#include <iostream>
#include <stdlib.h>
#include<vector>
#include<sstream>
#include <string>

std::string get_step(const std::string&, int, int min=0, int max=100);

int main()
{
    // TODO: Add input file argument
    // TODO: Add output file option
    // TODO: Add number of classes option
    // TODO: Add "help" option
    // TODO: Add "version" option

    const int ncls = 3;
    std::string line;
    std::getline(std::cin, line);
    std::cout << line << std::endl;

    while(!std::getline(std::cin, line).eof()) {
        std::stringstream sstream(line);
        std::string segment;
        std::vector<std::string> seglist;
        // FIXME: We could have more columns: use the first line to know the
        //        actual length of the array
        seglist.reserve(1235);

        while(std::getline(sstream, segment, ',')) {
            seglist.push_back(segment);
        }

        // FIXME: instead of hard-coded position, use the column names
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
