rm(list = ls())

library(shiny)
library(ggplot2)

# Inner module setup ==========================================================
inner_module_ui <- function(id) {
  ns <- NS(id)
  
  tagList(
    plotOutput(ns("inner_plot"))
  )
}

inner_module <- function(input, output, session) {
  # preallocate object
  string_return <- character(1)

  plot_object <- reactive({
    data <- data.frame(x = runif(10), y = runif(10))
    ggplot(data = data, aes(x = x, y = y)) +
      geom_point() +
      labs(title = "I am inside the inner module")
  })
  
  output$inner_plot <- renderPlot({plot_object()})
  
  return(isolate(plot_object()))
}

# Outer module setup ==========================================================
outer_module_ui <- function(id) {
  ns <- NS(id)
  
  fluidPage(
    inner_module_ui(ns("plot")),
    plotOutput(ns("text"))
  )
}

outer_module <- function(input, output, session) {
  inner_plot_return <- callModule(inner_module, "plot")

  output$text <- renderPlot({
    inner_plot_return + geom_line(colour = "red") +
      labs(title = "I am inside the outer module")
  })
}

# Shiny App functions =========================================================
ui <- fluidPage(
  outer_module_ui("outer")
)

server <- function(input, output, session) {
  callModule(outer_module, "outer")
}

shinyApp(ui, server)
