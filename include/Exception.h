//
// Created by marco on 19-8-13.
//

#ifndef PROJECT_EXCEPTION_H
#define PROJECT_EXCEPTION_H

#include <stdexcept>

class HPEException : public std::exception
{
public:
    HPEException(std::string message);
    virtual const char *what();

private:
    std::string m_message;
};

#endif //PROJECT_EXCEPTION_H
