import  numpy as np ;

def add_I(qlisp_ins, circ, q0):
    qlisp_ins.append('I', q0)
    circ.id(q0)

def add_U3(qlisp_ins, circ,  q0 ,th, phi, la) :
    #qlisp_ins.append( (('U' , th , phi , la) , q0) ) ;
    qlisp_ins.append((("Rz" , la) , q0));
    qlisp_ins.append((("Ry" , th) , q0)); 
    qlisp_ins.append((("Rz" , phi) , q0));  

    circ.rz(q0 , la) ; 
    circ.ry(q0 , th) ; 
    circ.rz(q0 , phi) ;



def add_CZ(qlisp_ins, circ,  q0 ,q1) :
    qlisp_ins.append(  ('CZ' ,  (q0 ,q1)) ) ;
    circ.cz(q0,q1);


def add_H(qlisp_ins, circ, q0):
    qlisp_ins.append(('H', q0))

    circ.h(q0)


def add_cnot(qlisp_ins, circ, q0, q1):
    qlisp_ins.append(('H', q1))
    qlisp_ins.append(  ('CZ' ,  (q0 ,q1)) )
    qlisp_ins.append(('H', q1))

    circ.h(q1)
    circ.cz(q0,q1)
    circ.h(q1)

def add_R(axis:str, qlisp_ins, circ, phi, q0):
    if axis == 'Rx':
        qlisp_ins.append((('Rx', phi), q0))
        circ.rx(q0,phi)
    if axis == 'Ry':
        qlisp_ins.append((('Ry', phi), q0))
        circ.ry(q0,phi)
    if axis == 'Rz':
        qlisp_ins.append((('Rz', phi), q0))
        circ.rz(q0,phi)


def swap(qlisp_ins, circ, q0 , q1) :
    # qlisp_ins.append([("Barrier", [0,1,2,3])])
    # circ.barrier([0,1,2,3])
    add_U3(qlisp_ins,circ ,  q1  , -np.pi/2 , 0 ,0 ) ;
    add_CZ(qlisp_ins, circ ,q0 , q1);
    add_U3(qlisp_ins,circ ,  q0  , -np.pi/2 , 0 ,0 ) ;
    add_U3(qlisp_ins,circ ,  q1  , np.pi/2 , 0 ,0 ) ; 
    add_CZ(qlisp_ins, circ ,q0 , q1);
    add_U3(qlisp_ins,circ ,  q0  , np.pi/2 , 0 ,0 ) ;
    add_U3(qlisp_ins,circ ,  q1  , -np.pi/2 , 0 ,0 ) ;
    add_CZ(qlisp_ins, circ ,q0 , q1);
    add_U3(qlisp_ins,circ ,  q1  , np.pi/2 , 0 ,0 ) ;

def add_barrier(qlisp_ins, circ, qlist):
    qlisp_ins.append(("Barrier", qlist))
    circ.barrier()

def add_measure(qlisp_ins, circ, qlist):
    for i in range(len(qlist)):
        qlisp_ins.append((('Measure', i), qlist[i]))

    circ.measure(pos = qlist)



########################## matrix and decompose #########################
def mat_r(axis, theta):    
    if axis == 'Rz': 
        mat = np.array([[np.exp(-1j*theta/2),0],[0,np.exp(1j*theta/2)]])
    if axis == 'Rx': 
        mat = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]])
    return mat

def mat_h():
    mat_h = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
    return mat_h














