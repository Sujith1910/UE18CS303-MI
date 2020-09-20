import sys


def pop_frontier_heu(frontier):
    if len(frontier) == 0:
        return None
    frontier.sort()
    max_values = []
    heuristic = []
    min = frontier[0][0]
    for key,cost, path in frontier:
        if key == min:
            min = key
            max_values.append(path)
            heuristic.append(cost)
        elif key > min:
            break

    max_values = sorted(max_values, key=lambda x: x[-1])
    # max_values.sort()
    desired_value = max_values[0]
    for key,cost, path in frontier:
        if path == desired_value:
            frontier.remove((key, cost, path))
            path_cost = cost
            break
    return path_cost, desired_value

def get_frontier_params_new_heu(node, frontier):
    for i in range(len(frontier)):
        curr_tuple = frontier[i]
        heu, cost, path = curr_tuple
        if path[-1] == node:
            return True, i, heu, cost, path

    return False, None, None, None, None


def A_star_Traversal(cost, heuristic, start_point, goals):
    path = []    
    explored_nodes = list()    
    
    for i in goals:
        if start_point == i:  
            path.append(start_point)
            return path  
        
    path.append(start_point)    
    path_cost = 0
    heuristic_cost = heuristic[start_point]
    frontier = [(heuristic_cost, path_cost, path)]    
    #print(frontier)
    
    while len(frontier) > 0:   
        path_cost_till_now, path_till_now = pop_frontier_heu(frontier)    
        current_node = path_till_now[-1]    
        explored_nodes.append(current_node)    
        #print("explored nodes are")
        #print(explored_nodes)
        #print(path_cost_till_now)
        #print(path_till_now)
        
        for i in goals:
            if current_node == i:    
                return path_till_now    
        
        neighbours = cost[current_node]    
    
        neighbours_list_int = [int(n) for n in neighbours]    
        #neighbours_list_int.sort(reverse=False)    
       # neighbours_list_str = [str(n) for n in neighbours_list_int]    
        #print(neighbours_list_int)
        
        i =0
        
        for neighbour in neighbours_list_int:
            if(neighbour > 0):              
                path_to_neighbour = path_till_now.copy()  
                path_to_neighbour.append(i)    
                #print(path_to_neighbour)
    
                extra_cost = neighbour
                neighbour_cost = extra_cost + path_cost_till_now
                heuristic_cost = neighbour_cost + heuristic[i]
                #print(heuristic_cost)
        
            
                new_element = (heuristic_cost, neighbour_cost, path_to_neighbour) 
                #print(new_element)
    
                
                is_there, indexx, neighbour_old_hue, neighbour_old_cost, _ = get_frontier_params_new_heu(i, frontier)    
        
                if (i not in explored_nodes) and not is_there:    
                    frontier.append(new_element)   
                elif is_there:    
                    if neighbour_old_hue > heuristic_cost:    
                        frontier.pop(indexx)    
                        frontier.append(new_element)
                #print(frontier)
            
            i+=1
    
    return None 


def pop_frontier(frontier):
    if len(frontier) == 0:
        return None
    min = sys.maxsize
    max_values = []
    for key, path in frontier:
        if key == min:
            max_values.append(path)
        elif key < min:
            min = key
            max_values.clear()
            max_values.append(path)

    max_values = sorted(max_values, key=lambda x: x[-1])
    # max_values.sort()
    desired_value = max_values[0]
    frontier.remove((min, max_values[0]))
    return min, desired_value


def get_frontier_params_new(node, frontier):
    for i in range(len(frontier)):
        curr_tuple = frontier[i]
        cost, path = curr_tuple
        if path[-1] == node:
            return True, i, cost, path

    return False, None, None, None



def UCS_Traversal(cost, start_point, goals):
    l = []
    path = []    
    explored_nodes = list()    
  
    for i in goals:
        if start_point == i:  
            path.append(start_point)
            return path    
    
    path.append(start_point)    
    path_cost = 0 
    frontier = [(path_cost, path)]    
    #print(frontier)
    
    while len(frontier) > 0:   
        path_cost_till_now, path_till_now = pop_frontier(frontier)    
        current_node = path_till_now[-1]    
        explored_nodes.append(current_node)    
       # print("explored nodes are")
        #print(explored_nodes)
        
        for i in goals:
            if current_node == i:    
                return path_till_now    
    
        neighbours = cost[current_node]    
    
        neighbours_list_int = [int(n) for n in neighbours]    
        #neighbours_list_int.sort(reverse=False)    
        #neighbours_list_str = [str(n) for n in neighbours_list_int]    
        #print(neighbours_list_str)
        i =0
        for neighbour in neighbours_list_int:
            if(neighbour > 0):              
                path_to_neighbour = path_till_now.copy()  
                path_to_neighbour.append(i)    
                #print(path_to_neighbour)
    
                extra_cost = neighbour
                neighbour_cost = extra_cost + path_cost_till_now    
                new_element = (neighbour_cost, path_to_neighbour) 
                #print(new_element)
                
                is_there, indexx, neighbour_old_cost, _ = get_frontier_params_new(i, frontier)    
        
                if (i not in explored_nodes) and not is_there:    
                    frontier.append(new_element)   
                elif is_there:    
                    if neighbour_old_cost > neighbour_cost:    
                        frontier.pop(indexx)    
                        frontier.append(new_element)
                #print(frontier)
            i+=1
    
    return None 



    return l

def DFS_Traversal(cost, start_point, goals):
    l = []
    t1=[start_point]
    first=True
    visited=set()

    while t1:
        node=t1.pop()
        print(node)
        if first==True:
            t1.append(start_point)
            first=False
        if node not in visited:
            visited.add(node)
            print("Visted:")
            print(visited)
            if node in goals:
                t1.append(node)
                break
            i = 0
            for val in cost[node]:
                if val>0:
                    if cost[node].index(val) in goals:
                        t1.append(i)
                        break
                    if cost[node].index(val) not in visited:
                        t1.append(i)
                i+=1
                print(t1)

    l = t1
    return l



'''
Function tri_Traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''



def tri_Traversal(cost, heuristic, start_point, goals):
    l = []

    #t1 = DFS_Traversal(cost, start_point, goals)
    t2 = UCS_Traversal(cost, start_point, goals)
    t3 = A_star_Traversal(cost, heuristic, start_point, goals)

    print(t2)
    print(t3)
    # t1 <= dfs traversal
    # t2 <= ucs	
    # t3 <= A_star_Traversal
    l.append(t2)
    l.append(t2)
    l.append(t3)
    print(l)
    return l

