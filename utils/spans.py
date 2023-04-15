def contained_in(outside: tuple[int,int], inside: tuple[int,int]):
    return (outside[0] <= inside[0]) and \
           (inside[1] <= outside[1])
