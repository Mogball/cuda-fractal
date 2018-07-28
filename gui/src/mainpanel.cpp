#include "mainpanel.h"
#include "mainwindow.h"
#include "display.h"

#include <QVBoxLayout>
#include <memory>

using namespace std;

class MainPanel::Impl {
public:
    Impl(MainPanel *pan);

private:
    unique_ptr<Display> m_display;
    unique_ptr<QVBoxLayout> m_layout;
};

MainPanel::Impl::Impl(MainPanel *pan) :
    m_display(make_unique<Display>(pan)),
    m_layout(make_unique<QVBoxLayout>()) {
    m_layout->addWidget(m_display.get());
    m_layout->setAlignment(Qt::AlignHCenter);
    pan->setLayout(m_layout.get());
}

MainPanel::MainPanel(MainWindow *win) :
    QWidget(win),
    m_impl(make_unique<Impl>(this)) {}

MainPanel::~MainPanel() = default;
