//
// Created by marco on 19-8-13.
//

#include "Exception.h"

HPEException::HPEException(std::string message)
    : m_message(message)
{
}

const char *HPEException::what()
{
    return m_message.c_str();
}