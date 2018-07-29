#include "control.h"
#include "mainpanel.h"

using namespace std;

class Control::Impl {
public:
    Impl(Control *control);
};

Control::Impl::Impl(Control *control) {
}

Control::Control(MainPanel *panel) :
    QWidget(panel),
    m_impl(make_unique<Impl>(this)) {}

Control::~Control() = default;
