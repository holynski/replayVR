#include "replay/util/strings.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace replay {

std::string GetLine(std::ifstream* filestream) {
  std::string line;
  getline(*filestream, line);
  // Remove carriage returns
  line = ReplaceAll(line, "\r", "");
  return line;
}

std::string ReplaceAll(const std::string& input, const std::string& search, const std::string& replace) {
  std::string output = input;
  size_t pos = 0;
  while ((pos = output.find(search, pos)) != std::string::npos) {
    output.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return output;
}

std::string PadZeros(const int input, const int length) {
  std::stringstream ss;
  ss.fill('0');
  ss.width(length);
  ss << input;
  return ss.str();
}

std::vector<std::string> Tokenize(const std::string& input,
                                  const char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(input);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

}  // namespace replay
