#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace cv;
using namespace std;

// ============================================================
// Config - все настройки и константы в одном месте
// ============================================================
namespace Config {
    // Ширина человека (плечи + руки в спокойном состоянии) в среднем около 0.45 - 0.5 метра.
    constexpr float REAL_PERSON_WIDTH = 0.5f;

    // Дефолтное фокусное расстояние (будет перезаписано после калибровки)
    constexpr float DEFAULT_FOCAL_LENGTH = 0.0028f;

    // Размер пикселя матрицы (используется в формуле расстояния и в калибровке)
    constexpr double PIXEL_SIZE = 0.0000085;

    // Калибровка
    constexpr float CALIBRATION_DISTANCE = 2.0f;   // встаём ровно в 2 метрах
    constexpr int   CALIBRATION_SECONDS = 5;      // секунд на подготовку

    // Сглаживание расстояния (EMA): 85% старого значения + 15% нового
    constexpr float EMA_OLD_WEIGHT = 0.85f;
    constexpr float EMA_NEW_WEIGHT = 0.15f;

    // Детектор
    constexpr int   PERSON_CLASS_ID = 15;
    constexpr float DETECTION_THRESHOLD = 0.5f;

    // Камера
    constexpr int    MAX_CAMERAS = 4;        // перебираем индексы 0..3
    constexpr int    FRAME_WIDTH = 1280;
    constexpr int    FRAME_HEIGHT = 720;
    constexpr int    MAX_BAD_FRAMES = 60;       // столько "серых" кадров подряд -> переподключение
    constexpr double MIN_FRAME_STDDEV = 2.0;    // кадр с меньшим разбросом цветов считаем серым/пустым

    // Значение фокуса после отключения автофокуса.
    // -1 = не трогать (останется положение объектива на момент отключения).
    // Нужное число смотрите в консоли в строке "фокус до" при первом запуске.
    constexpr int FIXED_FOCUS = -1;

    constexpr const char* WINDOW_NAME = "Distance Estimator";
}

// ============================================================
// Detection - один найденный объект
// ============================================================
struct Detection {
    Rect  box;
    float confidence = 0.0f;
    int   classId = -1;
};

// ============================================================
// CameraSource - поиск, открытие и чтение кадров с веб-камеры
// ============================================================
class CameraSource {
public:
    enum class ReadStatus { Ok, BlankFrame };

    // Ищет рабочую камеру, начиная с индекса startFrom
    bool open(int startFrom = 0) {
        return findCamera(startFrom, Config::MAX_CAMERAS);
    }

    // Переключается на следующую камеру (по кругу)
    bool switchNext() {
        int next = index_ + 1;
        cap_.release();
        badFrames_ = 0;
        return findCamera(next, Config::MAX_CAMERAS);
    }

    // Читает кадр. Серые/пустые кадры помечает как BlankFrame,
    // а если их слишком много подряд - переоткрывает ту же камеру.
    ReadStatus read(Mat& out) {
        Mat raw;
        if (cap_.isOpened()) cap_ >> raw;

        if (isGoodFrame(raw)) {
            badFrames_ = 0;
            out = raw.clone();   // рисуем на копии, а не в буфере камеры
            return ReadStatus::Ok;
        }

        out = raw.empty() ? Mat() : raw.clone();

        if (++badFrames_ >= Config::MAX_BAD_FRAMES) {
            badFrames_ = 0;
            cout << "Камера не отдаёт нормальные кадры, переподключаюсь..." << endl;
            cap_.release();
            findCamera(index_ < 0 ? 0 : index_, 1);   // только та же камера
        }
        return ReadStatus::BlankFrame;
    }

    void release() { cap_.release(); }

private:
    VideoCapture cap_;
    int index_ = -1;
    int badFrames_ = 0;

    // Кадр годится, если он не пустой и не однотонный (серая заглушка)
    static bool isGoodFrame(const Mat& f) {
        if (f.empty()) return false;
        Scalar mean, stddev;
        meanStdDev(f, mean, stddev);
        return (stddev[0] + stddev[1] + stddev[2]) > Config::MIN_FRAME_STDDEV;
    }

    // Ждём, пока камера «прогреется» и выдаст нормальный кадр (до ~1.6 сек)
    bool warmUp() {
        Mat f;
        for (int k = 0; k < 40; k++) {
            cap_ >> f;
            if (isGoodFrame(f)) return true;
            this_thread::sleep_for(chrono::milliseconds(40));
        }
        return false;
    }

    void applyFocusSettings() {
        cout << "Автофокус до: " << cap_.get(CAP_PROP_AUTOFOCUS)
            << ", фокус до: " << cap_.get(CAP_PROP_FOCUS) << endl;

        cap_.set(CAP_PROP_AUTOFOCUS, 0);   // выключаем автофокус
        if (Config::FIXED_FOCUS >= 0) cap_.set(CAP_PROP_FOCUS, Config::FIXED_FOCUS);

        cout << "Автофокус после: " << cap_.get(CAP_PROP_AUTOFOCUS)
            << ", фокус после: " << cap_.get(CAP_PROP_FOCUS) << endl;
    }

    bool tryOpen(int index, int backend) {
        // Сначала MJPG, потом формат по умолчанию: на части камер
        // MJPG даёт серый экран, на части - наоборот только он и работает.
        const bool formats[] = { true, false };

        for (bool mjpg : formats) {
            cap_.open(index, backend);
            if (!cap_.isOpened()) return false;

            // Формат и разрешение задаём СРАЗУ после открытия
            if (mjpg) cap_.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap_.set(CAP_PROP_FRAME_WIDTH, Config::FRAME_WIDTH);
            cap_.set(CAP_PROP_FRAME_HEIGHT, Config::FRAME_HEIGHT);

            if (warmUp()) {
                applyFocusSettings();
                cout << "Формат: " << (mjpg ? "MJPG" : "по умолчанию") << endl;
                return true;
            }
            cap_.release();
        }
        return false;
    }

    bool findCamera(int startFrom, int count) {
        const int backends[] = { CAP_DSHOW, CAP_MSMF };
        for (int i = 0; i < count; i++) {
            int idx = (startFrom + i) % Config::MAX_CAMERAS;
            for (int backend : backends) {
                cout << "Пробую камеру " << idx << ", бэкенд " << backend << "..." << endl;
                if (tryOpen(idx, backend)) {
                    index_ = idx;
                    cout << "[УСПЕХ] Камера " << idx << ", "
                        << cap_.get(CAP_PROP_FRAME_WIDTH) << "x"
                        << cap_.get(CAP_PROP_FRAME_HEIGHT) << endl;
                    return true;
                }
            }
        }
        return false;
    }
};

// ============================================================
// ObjectDetector - нейросеть MobileNet-SSD
// По умолчанию ищет только людей; чтобы отслеживать другие
// объекты, передайте их id классов (см. CLASS_NAMES).
// ============================================================
class ObjectDetector {
public:
    // Классы модели
    static const vector<string>& classNames() {
        static const vector<string> names = {
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        };
        return names;
    }

    ObjectDetector(const string& prototxt, const string& weights,
        vector<int> targetClasses = { Config::PERSON_CLASS_ID },
        float threshold = Config::DETECTION_THRESHOLD)
        : targetClasses_(move(targetClasses)), threshold_(threshold) {
        try {
            net_ = dnn::readNetFromCaffe(prototxt, weights);
            // net_.setPreferableBackend(dnn::DNN_BACKEND_DEFAULT);
            // net_.setPreferableTarget(dnn::DNN_TARGET_CPU);
            // Поддержка видеокарты AMD Radeon (OpenCL):
            // net_.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
            // net_.setPreferableTarget(dnn::DNN_TARGET_OPENCL);
        }
        catch (const cv::Exception&) {
            throw runtime_error("Убедитесь, что файлы .prototxt и .caffemodel "
                "лежат в папке с программой.");
        }
    }

    vector<Detection> detect(const Mat& frame) {
        Mat blob = dnn::blobFromImage(frame, 0.007843, Size(300, 300),
            Scalar(127.5, 127.5, 127.5), false, false);
        net_.setInput(blob);
        Mat out = net_.forward();
        Mat det(out.size[2], out.size[3], CV_32F, out.ptr<float>());

        vector<Detection> result;
        for (int i = 0; i < det.rows; i++) {
            float confidence = det.at<float>(i, 2);
            if (confidence <= threshold_) continue;

            int classId = (int)det.at<float>(i, 1);
            if (!isTarget(classId)) continue;

            int x1 = (int)(det.at<float>(i, 3) * frame.cols);
            int y1 = (int)(det.at<float>(i, 4) * frame.rows);
            int x2 = (int)(det.at<float>(i, 5) * frame.cols);
            int y2 = (int)(det.at<float>(i, 6) * frame.rows);

            Rect box(x1, y1, x2 - x1, y2 - y1);
            if (box.width <= 0 || box.height <= 0) continue;

            result.push_back({ box, confidence, classId });
        }
        return result;
    }

private:
    dnn::Net net_;
    vector<int> targetClasses_;
    float threshold_;

    bool isTarget(int classId) const {
        return find(targetClasses_.begin(), targetClasses_.end(), classId) != targetClasses_.end();
    }
};

// ============================================================
// ExponentialFilter - сглаживание значений (EMA), убирает «прыжки»
// ============================================================
class ExponentialFilter {
public:
    ExponentialFilter(float oldWeight, float newWeight)
        : oldWeight_(oldWeight), newWeight_(newWeight) {
    }

    float apply(float x) {
        if (value_ < 0) {
            value_ = x;   // инициализация при первом значении
        }
        else {
            value_ = (value_ * oldWeight_) + (x * newWeight_);
        }
        return value_;
    }

    void reset() { value_ = -1.0f; }

private:
    float oldWeight_;
    float newWeight_;
    float value_ = -1.0f;
};

// ============================================================
// DistanceEstimator - расчёт расстояния по ширине рамки
// ============================================================
class DistanceEstimator {
public:
    DistanceEstimator(float realWidth = Config::REAL_PERSON_WIDTH,
        float focalLength = Config::DEFAULT_FOCAL_LENGTH)
        : realWidth_(realWidth),
        focalLength_(focalLength),
        filter_(Config::EMA_OLD_WEIGHT, Config::EMA_NEW_WEIGHT) {
    }

    // Формула расстояния (без изменений)
    float rawDistance(float pixelWidth) const {
        return static_cast<float>((realWidth_ * focalLength_) / (pixelWidth * Config::PIXEL_SIZE));
    }

    // Расстояние после сглаживания
    float smoothedDistance(float pixelWidth) {
        return filter_.apply(rawDistance(pixelWidth));
    }

    // Калибровка (формула без изменений):
    // F = (ширина в пикселях * размер пикселя * известное расстояние) / реальная ширина
    void calibrate(float pixelWidth, float knownDistance) {
        focalLength_ = static_cast<float>((pixelWidth * Config::PIXEL_SIZE * knownDistance) / realWidth_);
        filter_.reset();
    }

    float focalLength() const { return focalLength_; }

private:
    float realWidth_;
    float focalLength_;
    ExponentialFilter filter_;
};

// ============================================================
// CalibrationController - конечный автомат калибровки
// ============================================================
class CalibrationController {
public:
    enum class State { Normal, Countdown, Capture };

    State state() const { return state_; }
    bool  isNormal() const { return state_ == State::Normal; }
    bool  isCountdown() const { return state_ == State::Countdown; }

    void start() {
        state_ = State::Countdown;
        startTime_ = chrono::steady_clock::now();
        cout << "Запуск таймера калибровки..." << endl;
    }

    int secondsLeft() const {
        auto now = chrono::steady_clock::now();
        int passed = (int)chrono::duration_cast<chrono::seconds>(now - startTime_).count();
        return Config::CALIBRATION_SECONDS - passed;
    }

    // Вызывается каждый кадр после детекции
    void update(const vector<Detection>& people, DistanceEstimator& estimator) {
        if (state_ == State::Countdown) {
            if (secondsLeft() <= 0) state_ = State::Capture;
        }
        else if (state_ == State::Capture) {
            // Ищем самого уверенного человека в кадре
            const Detection* best = nullptr;
            for (const auto& d : people) {
                if (!best || d.confidence > best->confidence) best = &d;
            }

            if (best) {
                estimator.calibrate((float)best->box.width, Config::CALIBRATION_DISTANCE);
                cout << "Калибровка успешна! Новое фокусное расстояние: "
                    << estimator.focalLength() << endl;
            }
            else {
                cout << "Ошибка калибровки: человек не найден в кадре." << endl;
            }
            state_ = State::Normal;
        }
    }

private:
    State state_ = State::Normal;
    chrono::steady_clock::time_point startTime_;
};

// ============================================================
// Renderer - вся отрисовка в одном месте
// ============================================================
class Renderer {
public:
    static void drawPerson(Mat& frame, const Detection& d, float distanceMeters) {
        ostringstream ss;
        ss << "Person: " << fixed << setprecision(2) << distanceMeters << "m";
        string label = ss.str();

        rectangle(frame, d.box, Scalar(0, 255, 0), 2);

        int baseLine = 0;
        Size ls = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
        int x = d.box.x;
        int minY = ls.height + 5;                 // подпись не уезжает за верх кадра
        int y = d.box.y > minY ? d.box.y : minY;

        rectangle(frame, Point(x, y - ls.height - 5), Point(x + ls.width, y),
            Scalar(0, 255, 0), FILLED);
        putText(frame, label, Point(x, y - 2), FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(0, 0, 0), 2);
    }

    static void drawHint(Mat& frame) {
        putText(frame, "C - calibrate (stand 2m away) | N - next camera | Q - quit",
            Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
    }

    static void drawCountdown(Mat& frame, int secondsLeft) {
        putText(frame, "CALIBRATION: Stand exactly 2 meters away!", Point(50, 50),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        putText(frame, "Time left: " + to_string(secondsLeft) + " sec", Point(50, 100),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
    }

    static void drawNoSignal(Mat& frame) {
        putText(frame, "No picture from camera... (N - next camera, Q - quit)",
            Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    }
};

// ============================================================
// App - связывает всё вместе и крутит главный цикл
// ============================================================
class App {
public:
    App()
        : detector_("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel") {
    }

    int run() {
        cout << "Ищем доступную камеру..." << endl;
        if (!camera_.open(0)) {
            cerr << "[ОШИБКА] Не удалось найти рабочую камеру!" << endl;
            cerr << "Проверьте:" << endl;
            cerr << "1. Шторку приватности на самой камере." << endl;
            cerr << "2. Кнопку отключения камеры на клавиатуре." << endl;
            cerr << "3. Настройки конфиденциальности Windows (разрешён ли доступ)." << endl;
            cerr << "4. Что камеру не держит другое приложение (Teams, Zoom, Telegram, браузер)." << endl;
            return -1;
        }

        printHelp();

        Mat frame;
        while (true) {
            // Серый/пустой кадр: показываем сообщение, детекцию пропускаем
            if (camera_.read(frame) == CameraSource::ReadStatus::BlankFrame) {
                if (!frame.empty()) {
                    Renderer::drawNoSignal(frame);
                    imshow(Config::WINDOW_NAME, frame);
                }
                else {
                    this_thread::sleep_for(chrono::milliseconds(30));
                }
                if (!handleKey(waitKey(30), false)) break;
                continue;
            }

            vector<Detection> people = detector_.detect(frame);

            if (calibration_.isNormal()) {
                for (const auto& p : people) {
                    float dist = estimator_.smoothedDistance((float)p.box.width);
                    Renderer::drawPerson(frame, p, dist);
                }
                Renderer::drawHint(frame);
            }
            else if (calibration_.isCountdown()) {
                int left = calibration_.secondsLeft();
                if (left > 0) Renderer::drawCountdown(frame, left);
            }

            calibration_.update(people, estimator_);

            imshow(Config::WINDOW_NAME, frame);

            if (!handleKey(waitKey(30), true)) break;
        }

        camera_.release();
        destroyAllWindows();
        return 0;
    }

private:
    CameraSource camera_;
    ObjectDetector detector_;
    DistanceEstimator estimator_;
    CalibrationController calibration_;

    // false - пора выходить
    bool handleKey(int key, bool canCalibrate) {
        if (key == 'q' || key == 'Q' || key == 27) return false;

        if ((key == 'c' || key == 'C') && canCalibrate && calibration_.isNormal()) {
            calibration_.start();
        }
        else if (key == 'n' || key == 'N') {
            if (!camera_.switchNext()) {
                cerr << "[ОШИБКА] Нет рабочей камеры." << endl;
                return false;
            }
        }
        return true;
    }

    static void printHelp() {
        cout << "=========================================" << endl;
        cout << "УПРАВЛЕНИЕ:" << endl;
        cout << "Клавиша 'C' - Начать калибровку камеры." << endl;
        cout << "Клавиша 'N' - Переключить камеру." << endl;
        cout << "Клавиша 'Q' - Выход из программы." << endl;
        cout << "=========================================" << endl;
    }
};

// ============================================================
int main() {
    // Включаем поддержку русского языка в консоли
    setlocale(LC_ALL, "Russian");

    int code = 0;
    try {
        App app;
        code = app.run();
    }
    catch (const exception& e) {
        cerr << "[ОШИБКА] " << e.what() << endl;
        code = -1;
    }

    if (code != 0) system("pause");   // чтобы консоль не закрылась сразу
    return code;
}
