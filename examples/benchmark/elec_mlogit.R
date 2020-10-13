#install.packages("mlogit")

suppressMessages(library(mlogit))

df = read.csv("https://raw.githubusercontent.com/arteagac/mixedlogit/master/examples/data/electricity_long.csv")

Electr <- mlogit.data(df, shape="long", id.var="id", chid.var="chid", choice="choice", alt.var="alt")
set.seed(0)
cat("\n\n=== Electricity dataset. R::mlogit ===")
cat("\nNdraws Time(s) Log-Likeli. RAM(GB) GPU(GB) Converg.")

for(i in 1:5){
    Rprof(interval = 0.1, memory.profiling = TRUE)
    
    model = mlogit(choice~pf+cl+loc+wk+tod+seas|0, Electr,
            rpar=c(pf="n",cl="n",loc="n",wk="n",tod="n",seas="n"), 
            R=i*100,halton=NA,print.level=0,panel=TRUE)
            
    Rprof(append=TRUE)
    time = (summaryRprof("Rprof.out"))$by.total[1, "total.time"]
    sm = summaryRprof("Rprof.out", memory = "tseries", diff = FALSE)
    sm$tot = sm$vsize.small + sm$vsize.large + sm$nodes
    mem = max(sm$tot)/(1024*1024*1024)
    #Format and print output
    res = c()
    res = c(res, format(i*100, width = 6, justify = "right", trim=T), " ")
    res = c(res, format(time, width = 7, nsmall = 2, justify = "right", trim=T), " ")
    res = c(res, format(model$logLik[1], width = 11, nsmall = 2, justify = "right", trim=T), " ")
    res = c(res, format(mem, width = 4, nsmall = 3, justify = "right", trim=T), " ")
    res = c(res, format(0, width = 7, nsmall = 3, justify = "right", trim=T), " ")
    res = c(res, format(model$est.stat$code == 1, width = 5, justify = "left", trim=T), " ")
    cat("\n", res, sep="")
}

cat("\n")
