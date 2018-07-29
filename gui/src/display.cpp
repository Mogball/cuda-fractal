#include "image.h"
#include "display.h"
#include "mainpanel.h"

#include <QPainter>
#include <memory>

using namespace std;

static constexpr int display_dim = 2000;

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

void Display::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.drawImage(0, 0, m_impl->del->image());
    painter.end();
}

void Display::renderUpdate() {
    update();
}
