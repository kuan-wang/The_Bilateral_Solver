
#ifndef _UNIQUE_HPP_
#define _UNIQUE_HPP_

    void unique(std::vector<double>& hashed_coords, std::vector<double>& unique_hashes,
                std::vector<int>& unique_idx,std::vector<int>& idx)
    {
        std::set<double> input;
        std::cout << "for 1" << std::endl;
        std::cout << "hashed_coords size" <<hashed_coords.size()<< std::endl;
        for (int i = 0; i < hashed_coords.size(); i++) {
            // std::cout << "hashed_coords:"<<hashed_coords[i] << std::endl;
            input.insert(hashed_coords[i]);
        }
        unique_hashes.resize(input.size());
        unique_idx.resize(input.size(),-1);
        idx.resize(npixels);
        std::copy(input.begin(),input.end(),unique_hashes.begin());
        // std::cout << "input :" <<unique_hashes<< std::endl;
        std::cout << "input size" <<input.size()<< std::endl;

        std::cout << "for 2" << std::endl;
        for (int i = 0; i < hashed_coords.size(); i++) {
            int id = binarySearchRecursive(&unique_hashes[0],0,input.size(),hashed_coords[i]);
            if(id >= 0)
            {
                idx.push_back(id);
                if(unique_idx[id] < 0) unique_idx[id] = i;
            }
        }

        std::cout << "for 2 end" << std::endl;

    }


#endif //_UNIQUE_HPP_
