#include "mainwindow.h"
#include "mainpanel.h"

#include <QLabel>
#include <QLayout>
#include <memory>

static constexpr int window_width = 2020;
static constexpr int window_height = 2020;

using namespace std;

class MainWindow::Impl {
public:
    Impl(MainWindow *win);

private:
    unique_ptr<MainPanel> m_panel;
};

MainWindow::Impl::Impl(MainWindow *win) :
    m_panel(make_unique<MainPanel>(win)) {
    win->setCentralWidget(m_panel.get());
}

MainWindow::MainWindow() :
    m_impl(make_unique<Impl>(this)) {
    setFixedSize(window_width, window_height);
}

MainWindow::~MainWindow() = default;
