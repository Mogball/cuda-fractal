#include "mainwindow.h"

#include <QApplication>
#include <memory>

using namespace std;

static constexpr auto window_name = "CUDA Fractals";

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName(window_name);
    auto window = make_unique<MainWindow>();
    window->setWindowTitle(window_name);
    window->show();
    return app.exec();
}
