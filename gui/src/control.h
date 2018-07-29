#pragma once
#include <QWidget>
#include <memory>

class MainPanel;

class Control : public QWidget {
Q_OBJECT

public:
    Control(MainPanel *panel);
    ~Control();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
