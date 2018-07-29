#pragma once

#include <QOpenGLWidget>
#include <memory>

class QPaintEvent;
class QWheelEvent;
class QMouseEvent;
class MainPanel;

class Display: public QOpenGLWidget {
public:
    Display(MainPanel *panel);
    ~Display();

    void paintEvent(QPaintEvent *ev) override;
    void wheelEvent(QWheelEvent *ev) override;
    void mouseReleaseEvent(QMouseEvent *ev) override;

protected:
    Q_SLOT void renderUpdate();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
