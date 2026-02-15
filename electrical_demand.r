# Cargar librerías
library(dplyr)
library(leaps)
library(Matrix)
library(glmnet)
library(corrplot)
library(ggplot2)


df <- read.csv("", sep = ",", dec = ".")


summary(df$demanda_MWh)


hist(df$demanda_MWh,
     main = "Distribución de la Demanda Eléctrica (MWh)",
     xlab = "Demanda (MWh)",
     col = "steelblue", border = "white")


var.numericas <- df[, sapply(df, is.numeric)]
summary(var.numericas)


Matriz.correlacion <- cor(var.numericas, use = "complete.obs")
corrplot(Matriz.correlacion, method = "color", type = "upper", tl.cex = 0.7)


plot(df$poblacion, df$demanda_MWh,
     main = "Demanda eléctrica vs Población",
     xlab = "Población", ylab = "Demanda eléctrica (MWh)",
     col = "steelblue", pch = 16)
abline(lm(demanda_MWh ~ poblacion, data = df), col = "red", lwd = 2)


promedio_mes <- aggregate(demanda_MWh ~ mes, data = df, mean)
ggplot(promedio_mes, aes(x = factor(mes), y = demanda_MWh)) +
  geom_col(fill = "steelblue") +
  geom_line(aes(group = 1), color = "red") +
  geom_point(color = "red") +
  labs(title = "Demanda eléctrica promedio por mes", 
       x = "Mes", y = "Demanda (MWh)") +
  coord_cartesian(ylim = c(600000, NA))


df <- df %>%
  mutate(fecha = anio * 100 + mes) %>%
  arrange(fecha)

val  <- df[df$fecha %in% tail(unique(df$fecha), 24), ] 
entr <- df[!(df$fecha %in% tail(unique(df$fecha), 24)), ] 

entr <- subset(entr, select = -c(fecha))
val  <- subset(val, select = -c(fecha))



set.seed(42)
modelo <- lm(demanda_MWh ~ ., data = entr)
coefficients(modelo)


ajuste <- regsubsets(demanda_MWh ~ ., data = entr, method = "forward", nvmax = ncol(entr) - 1)
resumen <- summary(ajuste)


aux_BIC <- which.min(resumen$bic)
variables_BIC <- names(which(resumen$which[aux_BIC,] == TRUE))
Modelo.BIC <- lm(demanda_MWh ~ poblacion + indice_industrial + tasa_desempleo, data = entr)


aux_RA <- which.max(resumen$adjr2)
variables_RA <- names(which(resumen$which[aux_RA,] == TRUE))
Modelo.RA <- lm(demanda_MWh ~ poblacion, data = entr)


set.seed(42)
x.entr <- model.matrix(demanda_MWh ~ . , data = entr)[, -1]
y.entr <- entr$demanda_MWh
x.val  <- model.matrix(demanda_MWh ~ . , data = val)[, -1]

rl <- cv.glmnet(x.entr, y.entr, alpha = 1, nlambda = 200)



calc_R2 <- function(model, data, val_y) {
  pred <- predict(model, data)
  RSS <- sum((val_y - pred)^2)
  TSS <- sum((val_y - mean(val_y))^2)
  n <- length(val_y)
  d <- length(model$coefficients) - 1
  R2 <- 1 - ((RSS / (n - d - 1)) / (TSS / (n - 1)))
  return(list(MSE = mean((val_y - pred)^2), R2 = R2))
}


eval_BIC <- calc_R2(Modelo.BIC, val, val$demanda_MWh)
eval_RA  <- calc_R2(Modelo.RA, val, val$demanda_MWh)


predict.lasso <- predict(rl, x.val, s = rl$lambda.min)
RSS.LS <- sum((val$demanda_MWh - predict.lasso)^2)
TSS.LS <- sum((val$demanda_MWh - mean(val$demanda_MWh))^2)
n.LS   <- nrow(val)
coef_lasso <- coef(rl, s = rl$lambda.min)
d.LS   <- sum(coef_lasso[-1] != 0)
RA.LS  <- 1 - ((RSS.LS / (n.LS - d.LS - 1)) / (TSS.LS / (n.LS - 1)))
MSE.LS <- mean((val$demanda_MWh - predict.lasso)^2)

resultados <- data.frame(
  Modelo = c("BIC", "R2 Ajustado", "LASSO"),
  MSE = c(eval_BIC$MSE, eval_RA$MSE, MSE.LS),
  R2_Ajustado = c(eval_BIC$R2, eval_RA$R2, RA.LS)
)

print(resultados)
