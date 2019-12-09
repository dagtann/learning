rm(list = ls())
setwd("C:/Users/32856/Documents/git/learning/rshiny/modules/output_example")
packages <- c("shiny", "ggplot2")
lapply(packages, library, character.only = TRUE, quietly = TRUE)

source("linkedScatter.R")

ui <- fixedPage(
  h2("Module example"),
  linkedScatterUI("scatters"),
  textOutput("summary")
)

server <- function(input, output, session) {
  df <- callModule(linkedScatter, "scatters", reactive(mpg),
    left = reactive(c("cty", "hwy")), right = reactive(c("drv", "hwy"))
  )
  output$summary <- renderText({
    sprintf("%d observation(s) selected", nrow(dplyr::filter(df(), selected_)))
  })
}

shinyApp(ui, server)