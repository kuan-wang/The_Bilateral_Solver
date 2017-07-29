
#ifndef _BINARYSEARCH_HPP_
#define _BINARYSEARCH_HPP_

    template <typename T>
    int binarySearchRecursive(const T a[],int low,int high,T key){
        if(low>high)
            return -(low+1);

        int mid=low+(high-low)/2;
        if(key<a[mid])
            return binarySearchRecursive(a,low,mid-1,key);
        else if(key > a[mid])
            return binarySearchRecursive(a,mid+1,high,key);
        else
            return mid;

    }
    //
    // int binarySearchRecursive(const double a[],int low,int high,double key){
    //     if(low>high)
    //         return -(low+1);
    //
    //     int mid=low+(high-low)/2;
    //     if(key<a[mid])
    //         return binarySearchRecursive(a,low,mid-1,key);
    //     else if(key > a[mid])
    //         return binarySearchRecursive(a,mid+1,high,key);
    //     else
    //         return mid;
    //
    // }


#endif //_BISTOCHASTIZE_HPP_
