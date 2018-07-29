#include "image.h"
#include "display.h"
#include "mainpanel.h"

#include <QPainter>
#include <QPaintEvent>
#include <QWheelEvent>
#include <QMouseEvent>

using namespace std;

static constexpr int display_dim = 2000;

template<int lo, int hi>
static bool outside(int val) {
    return val < lo || val > hi;
}

template<int lo, int hi>
static bool inside(int val) {
    return lo <= val && val <= hi;
}

class Display::Impl {
public:
    Impl(int dim);

    unique_ptr<Image> del;
};

Display::Impl::Impl(int dim) :
    del(make_unique<Image>(dim)) {}

Display::Display(MainPanel *panel) :
    QOpenGLWidget(panel),
    m_impl(make_unique<Impl>(display_dim)) {
    setFixedSize(display_dim, display_dim);
    connect(
        m_impl->del.get(),
        &Image::renderDone,
        this,
        &Display::renderUpdate
    );
    m_impl->del->launchRender();
}

Display::~Display() = default;

void Display::paintEvent(QPaintEvent *ev) {
    QPainter painter(this);
    painter.drawImage(0, 0, m_impl->del->image());
    painter.end();
    ev->accept();
}

void Display::wheelEvent(QWheelEvent *ev) {
    QPoint numPixels = ev->pixelDelta();
    QPoint numDegrees = ev->angleDelta() / 8;
    int delta = 0;
    if (!numPixels.isNull()) {
        delta = numPixels.y();
    } else if (!numDegrees.isNull()) {
        delta = numDegrees.y() / 15;
    }
    if (delta != 0) {
        m_impl->del->zoom(delta);
        m_impl->del->launchRender();
    }
    ev->accept();
}

void Display::mouseReleaseEvent(QMouseEvent *ev) {
    if (outside<0, display_dim>(ev->x()) ||
        outside<0, display_dim>(ev->y())) {
        return;
    }
    m_impl->del->recenter(ev->x(), ev->y());
    m_impl->del->launchRender();
    ev->accept();
}

void Display::renderUpdate() {
    update();
}
