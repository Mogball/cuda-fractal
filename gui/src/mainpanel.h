#pragma once

#include <QWidget>
#include <memory>

class MainWindow;

class MainPanel : public QWidget {
Q_OBJECT

public:
    MainPanel(MainWindow *win);
    ~MainPanel();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

