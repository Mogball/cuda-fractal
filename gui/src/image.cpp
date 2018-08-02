#include "image.h"
#include <QTimer>

static constexpr long default_T = 1 << 12;
static constexpr double default_x = -0.5;
static constexpr double default_y = 0.0;
static constexpr double default_s = 3.0;

Image::Image(int dim) : m_vec(dim * dim), m_cfg() {
    m_cfg.max = dim;
    m_cfg.T = default_T;
    m_cfg.xc = default_x;
    m_cfg.yc = default_y;
    m_cfg.s = default_s;
    m_cfg.color = BlackGoldYellow;
    connect(this, &Image::imageDone, this, &Image::setImage);
}

QImage &Image::image() {
    return m_img;
}

void Image::launchRender() {
    size_t size = m_cfg.max * m_cfg.max;
    m_vec.reserve(size);
    QTimer::singleShot(0, this, &Image::renderImage);
}

void Image::recenter(int x, int y) {
    m_cfg.xc += (x - m_cfg.max / 2) * m_cfg.s / m_cfg.max;
    m_cfg.yc += (y - m_cfg.max / 2) * m_cfg.s / m_cfg.max;
}

void Image::zoom(int delta) {
    double zoom = delta / 10.0;
    if (delta > 0) {
        m_cfg.s /= zoom;
    } else {
        m_cfg.s *= -zoom;
    }
}

void Image::renderImage() {
    gpu_mandelbrot(m_cfg, m_vec.data());
    uchar *data = reinterpret_cast<uchar *>(m_vec.data());
    QImage img{data, m_cfg.max, m_cfg.max, QImage::Format_ARGB32};
    Q_EMIT imageDone(img);
}

void Image::setImage(QImage img) {
    m_img = img;
    Q_EMIT renderDone();
}
