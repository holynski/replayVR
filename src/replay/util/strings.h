#include <fstream>
#include <iostream>

namespace replay {

std::string GetLine(std::ifstream* filestream);

std::string ReplaceAll(const std::string& input, const std::string& search,
                       const std::string& replace);

std::string PadZeros(const int input, const int length);

// Given a string with multiple items delimited by a particular character,
// returns a vector of each of the items.
// E.g. ("this,is,the,input" => {"this", "is", "the", "input"})
// The delimiting character can be passed as the second argument
std::vector<std::string> Tokenize(const std::string& input,
                                  const char delimiter = ',');
}  // namespace replay
