#pragma once
#include <gpu.h>
#include <QObject>
#include <QImage>
#include <vector>
#include <cstdint>

struct render_config;

class Image : public QObject {
Q_OBJECT

public:
    Image(int dim);

    QImage &image();

    void launchRender();
    void recenter(int x, int y);
    void zoom(int delta);

protected:
    Q_SLOT void renderImage();
    Q_SLOT void setImage(QImage img);
    Q_SIGNAL void imageDone(QImage img);

public:
    Q_SIGNAL void renderDone();

private:
    std::vector<uint32_t> m_vec;
    render_config m_cfg;
    QImage m_img;
};
