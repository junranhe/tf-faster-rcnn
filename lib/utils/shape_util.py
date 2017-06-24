# shape util #
print_shape_flag=False
def print_shape(layer_weight,add_info=''):
    if(print_shape_flag):
        print(add_info+' layer {} shape is {}'.format(layer_weight.name,layer_weight.get_shape()))

