
#ifndef _GETVALIDIDX_HPP_
#define _GETVALIDIDX_HPP_




    void get_valid_idx(std::vector<double>& valid, std::vector<double>& candidates,
                        std::vector<int>& valid_idx, std::vector<int>& locs)
    {
        valid_idx.clear();
        locs.clear();
        for (int i = 0; i < candidates.size(); i++) {
            int id = binarySearchRecursive(&valid[0],0,candidates.size(),candidates[i]);
            if(id >= 0)
            {
                locs.push_back(id);
                valid_idx.push_back(i);
            }
        }
        // std::cout << "candidates.size()" << candidates.size() << std::endl;
        // std::cout << "valid_idx.size():"<< valid_idx.size() << std::endl;

    }



#endif //_GETVALIDIDX_HPP_
