#pragma once

#include <QOpenGLWidget>
#include <memory>

class QPaintEvent;
class MainPanel;

class Display: public QOpenGLWidget {
public:
    Display(MainPanel *panel);
    ~Display();

    void paintEvent(QPaintEvent *ev) override;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
