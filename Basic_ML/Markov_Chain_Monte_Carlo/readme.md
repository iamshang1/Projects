# Markov Chain Monte Carlo

This exercise utilizes the Markov Chain Monte Carlo algorithm to decrypt a segment of text that has gone through a simple substitution ciper, i.e., 
each letter has been substituted by a different letter. The encryped text segment is a small segment of text from the Hitchhiker's Guide to the Galaxy.

First, a large corpus of text is used to generate a 27x27 transition matrix that represents the probability that a letter will appear given the previous letter. Note that the 27x27 transition matrix
only includes alphabet letters and spaces; all other characters were removed from the text. For this exercise, the entire Hitchhiker's Guide to the Galaxy series was used to create the transition matrix.
Note that the encrypted text segment was not included in the creation of the transition matrix.

Next, we initialize a random mapping that maps substitute letters to actual letters. This mapping will be used to decode the encrypted text segment. The Markov Chain Monte Carlo algorithm is
applied to this random mapping and does the following:

 - Use the mapping to decode the encrypted text segment. Calculate the probability of the decrypted text segment using the transition matrix, i.e., multiply the probability of each letter in the decrypted
 text segment given the previous letter. This is the score of the mapping.
 - Randomly switch two letters in the mapping.
 - Use the new mapping to decode the encrypted text segment. Calculate the probability of the decrypted text segment using the transition matrix. This is the score of the new mapping.
 - If the score of the new mapping is higher than the score of the previous mapping, keep the new mapping.
 - If the score of the new mapping is less than the score of the previous mapping, keep the new mapping with probability (new map score/old map score). Otherwise keep the old mapping.
 - Repeat until convergence.
 
This algorithm is used on multiple random starting mappings until the correct cipher is found.

## Results

**Iteration 0:**
xrt xk fdd zxbbclcdctm tga ixyxw trvwap xw tga dcygt gcjbadk ga zcheap rz tga zcaha xk zfzav fyfcw fwp zdfhap f dcttda tche cw tga dcttda lxo qadd tgft qfb pxwa gcb bgcz bdrwe xkk cwtx tga cwem ixcp cw bzcta xk gficwy tfeaw qgft ga vayfvpap fb fw aotvajadm zxbctcia zcaha xk fhtcxw tga yvalrdxw dafpav awpap rz gficwy f iavm lfp jxwtg fktav fdd ct qfb zvattm jrhg tga bfja fb fdd tga zvaicxrb jxwtgb aohazt tgft tgava qfb wxq wxtgcwy xw tga tadaicbcxw fwm jxva ga zrt xw f dcttda dcygt jrbch cwbtafp

**Iteration 500:**
int im oss piffuzusute tha bidil tnrlay il tha sudht hugfasm ha puckay np tha puaca im popar odoul oly psocay o suttsa tuck ul tha suttsa zix wass thot wof yila huf fhup fsnlk imm ulti tha ulke biuy ul fputa im hobuld tokal whot ha radoryay of ol axtragase pifutuba puaca im octuil tha draznsil saoyar alyay np hobuld o bare zoy gilth omtar oss ut wof pratte gnch tha foga of oss tha prabuinf gilthf axcapt thot thara wof liw lithuld il tha tasabufuil ole gira ha pnt il o suttsa sudht gnfuc ulftaoy

**Iteration 2500:**
iut if all pissomoloty the vigin turned in the loght hobself he pocked up the poece if paper agaon and placed a lottle tock on the lottle mix well that was dine hos shop slunk iff onti the onky viod on spote if havong taken what he regarded as an extrebely pisotove poece if actoin the gremulin leader ended up havong a very mad binth after all ot was pretty buch the sabe as all the prevoius binths except that there was niw nithong in the televosoin any bire he put in a lottle loght busoc onstead
 
**Iteration 6000:**
out of all possibility the vogon turned on the light himself he picked up the piece of paper again and placed a little tick in the little box well that was done his ship slunk off into the inky void in spite of having taken what he regarded as an extremely positive piece of action the grebulon leader ended up having a very bad month after all it was pretty much the same as all the previous months except that there was now nothing on the television any more he put on a little light music instead
