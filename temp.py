out = [{'Label': 'person', 'confidence': 0.9995384812355042, 'X': 996, 'Y': 374, 'Width': 186, 'Height': 493, 'centerX': 1089, 'centerY': 621, 'right_below_x_point': 1182, 'right_below_y_point': 374}, {'Label': 'person', 'confidence': 0.9986192584037781, 'X': 852, 'Y': 387, 'Width': 177, 'Height': 494, 'centerX': 941, 'centerY': 634, 'right_below_x_point': 1029, 'right_below_y_point': 387}, {'Label': 'person', 'confidence': 0.9926905035972595, 'X': 763, 'Y': 404, 'Width': 116, 'Height': 393, 'centerX': 821, 'centerY': 601, 'right_below_x_point': 879, 'right_below_y_point': 404}, {'Label': 'car', 'confidence': 0.9763578176498413, 'X': -11, 'Y': 389, 'Width': 239, 'Height': 135, 'centerX': 108, 'centerY': 457, 'right_below_x_point': 227, 'right_below_y_point': 389}, {'Label': 'car', 'confidence': 0.961442768573761, 'X': 86, 'Y': 377, 'Width': 687, 'Height': 316, 'centerX': 430, 'centerY': 535, 'right_below_x_point': 773, 'right_below_y_point': 377}]


def numberingObjects(out):
    counter = dict()
    for d in out : 
        if d['Label'] not in counter : 
            counter[d['Label']] = 1 
            d['Label'] = d['Label'] + str(1)
        else : 
            counter[d['Label']] += 1 
            d['Label'] = d['Label'] + str(counter[d['Label']])
    return out, counter 

out, counter = numberingObjects(out)
print(out) 
print(counter)