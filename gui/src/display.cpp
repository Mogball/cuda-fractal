#include "display.h"
#include "mainpanel.h"

#include <QImage>
#include <QPainter>
#include <memory>

#include <QDebug>

using namespace std;

static constexpr int display_width = 1800;
static constexpr int display_height = 1200;

class Display::Impl {
public:
    Impl(Display *display);

    QImage &image();

private:
    QImage m_img;
};

Display::Impl::Impl(Display *display) :
    m_img(display_width, display_height, QImage::Format_ARGB32) {}

QImage &Display::Impl::image() { return m_img; }

Display::Display(MainPanel *panel) :
    QOpenGLWidget(panel),
    m_impl(make_unique<Impl>(this)) {
    setFixedSize(display_width, display_height);
}

Display::~Display() = default;

void Display::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.drawImage(0, 0, m_impl->image());
    painter.end();
}
