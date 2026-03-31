COMMENT

Na passive leak channel

ENDCOMMENT


NEURON {
   
	
	SUFFIX naleak
	USEION na READ ena
	RANGE gbar, i, ena
    NONSPECIFIC_CURRENT i  
}

PARAMETER {
	gbar = 0   	(S/cm2)
	

}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	i 	(mA/cm2)
    ena     (mV)
	v	(mV)
}
 

BREAKPOINT {

	i = gbar * (v - ena)
} 


