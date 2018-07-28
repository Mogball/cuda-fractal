#pragma once

#include <QMainWindow>
#include <memory>

class MainWindow : public QMainWindow {
Q_OBJECT

public:
    MainWindow();
    ~MainWindow();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
