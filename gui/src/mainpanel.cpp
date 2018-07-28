#include "mainpanel.h"
#include "mainwindow.h"

#include <QPushButton>
#include <QVBoxLayout>
#include <memory>

using namespace std;

class MainPanel::Impl {
public:
    Impl(MainPanel *pan);

private:
    unique_ptr<QVBoxLayout> m_layout;
    unique_ptr<QPushButton> m_label;
};

MainPanel::Impl::Impl(MainPanel *pan) :
    m_layout(make_unique<QVBoxLayout>()),
    m_label(make_unique<QPushButton>("Hello World")) {
    m_layout->addWidget(m_label.get());
    m_layout->setAlignment(Qt::AlignHCenter);
    pan->setLayout(m_layout.get());
}

MainPanel::MainPanel(MainWindow *win) :
    QWidget(win),
    m_impl(make_unique<Impl>(this)) {}

MainPanel::~MainPanel() = default;
