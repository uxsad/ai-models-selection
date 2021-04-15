#include <iostream>
#include <stdlib.h>
#include<vector>
#include<sstream>
#include <string>

std::string get_step(const std::string&, int, int min=0, int max=100);
int main()
{
    int ncls = 3;
    std::string line;
    std::getline(std::cin, line);
    std::cout << line << std::endl;

    while(!std::getline(std::cin, line).eof()) {
        std::stringstream sstream(line);
        std::string segment;
        std::vector<std::string> seglist;
        seglist.reserve(1235);

        size_t i = 0;
        while(std::getline(sstream, segment, ',')) {
            seglist.push_back(segment);
        }

        for(int i = 29; i <=38; ++i) {
          if(i==37)
            seglist[i] = get_step(seglist[i], ncls, -100, 100);
          else
            seglist[i] = get_step(seglist[i], ncls);
        }

        for(const auto &s : seglist) {
            std::cout << s;
            if(&s != &seglist.back())
                std::cout << ",";

        }
        std::cout<< std::endl;
    }

    return 0;
}

std::string get_step(const std::string& s, int cls, int min, int max)
{
  double MINIMUM = 2;
    if(s == "")
        return "";
    double val = atof(s.c_str());
    if(val < MINIMUM)
      return "";
    double width = (max - min) / cls;
    for(int i = 0; i < cls; ++i) {
        if ((min + width * i <= val) && (val < min + width * (i + 1)))
            return std::to_string(i);
    }
    if(val >= max)
        return std::to_string(cls - 1);
    return "";
}
