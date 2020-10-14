suppressMessages(library(mlogit))


# ======= ELECTRICTY DATASET ========
df = read.csv("https://raw.githubusercontent.com/arteagac/mixedlogit/master/examples/data/electricity_long.csv")

Electr <- mlogit.data(df, shape="long", id.var="id", chid.var="chid", choice="choice", alt.var="alt")
set.seed(0)
cat("\n\n=== Electricity dataset. R::mlogit ===")

model = mlogit(choice~pf+cl+loc+wk+tod+seas|0, Electr,
        rpar=c(pf="n",cl="n",loc="n",wk="n",tod="n",seas="n"), 
        R=600,halton=NA,print.level=0,panel=TRUE)

summ = coef(summary(model))
varnames = row.names(summ)
est_coeff = summ[,"Estimate"]
est_stder = summ[,"Std. Error"]
cat("\nVariable    Estimate   Std.Err.")
for (i in 1:nrow(summ)){
    cat("\n")
    cat(c(format(varnames[i], width=10, trim=T),
        format(est_coeff[i], width=10, nsmall = 5, digits = 5, trim=T),
            format(est_stder[i], width=9, nsmall = 5, digits = 5, trim=T)))
}
cat(c("\nLog.Lik:   ", format(logLik(model)[1], nsmall = 2, digits = 2, trim=T)))
cat("\n")
# ======= ARTIFICIAL DATASET ========
df = read.csv("https://raw.githubusercontent.com/arteagac/mixedlogit/master/examples/data/artificial_long.csv")

Artif <- mlogit.data(df, shape="long", id.var="id", choice="choice", alt.var="alt")
set.seed(0)
cat("\n\n=== Artificial dataset. R::mlogit ===")
model = mlogit(choice~price+time+conven+comfort+meals+petfr+emipp+nonsig1+nonsig2+nonsig3|0, Artif,
        rpar=c(meals="n", petfr="n", emipp="n"), 
        R=200,halton=NA,print.level=0)

summ = coef(summary(model))
varnames = row.names(summ)
est_coeff = summ[,"Estimate"]
est_stder = summ[,"Std. Error"]
cat("\nVariable    Estimate   Std.Err.")
for (i in 1:nrow(summ)){
    cat("\n")
    cat(c(format(varnames[i], width=10, trim=T),
        format(est_coeff[i], width=10, nsmall = 5, digits = 5, trim=T),
            format(est_stder[i], width=9, nsmall = 5, digits = 5, trim=T)))
}
cat(c("\nLog.Lik:   ", format(logLik(model)[1], nsmall = 2, digits = 2, trim=T)))
cat("\n")