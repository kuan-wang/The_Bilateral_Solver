
#ifndef _HASHCOORDS_HPP_
#define _HASHCOORDS_HPP_



    void hash_coords(std::vector<double>& coords_flat, std::vector<double>& hashed_coords)
    {
        double max_val = 255.0;
        hashed_coords.clear();
        for (int i = 0; i < coords_flat.size()/dim; i++) {
            double hash = 0;
            for (int j = 0; j < dim; j++) {
                hash += coords_flat[i*dim+j] + hash*max_val;
            }
            hashed_coords.push_back(hash);
        }
    }

#endif //_HASHCOORDS_HPP_
