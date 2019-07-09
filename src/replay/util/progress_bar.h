#pragma once

namespace replay {

void PrintProgress(const size_t current, const size_t maximum,
                   const std::string& pre_message = "",
                   const std::string& post_message = "");

}
