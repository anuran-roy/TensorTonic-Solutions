The trick here is to realize that the number of rows is always going to be one. Or the number of columns, but either one of the dimensions, so it's never going to be  a 2D matrix. It's always going to be a 1D vector. So that's why you have to do row = blockIdx.x and col equal to threadIdx.x. 



That's also what got my first submission wrong :P