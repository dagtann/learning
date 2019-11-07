library("shiny")

ui <- fluidPage(
    sliderInput(inputId = "num", label = "Set the number of samples",
                min = 10, max = 1000, value = 25, step = 1),
    plotOutput(outputId = "hist")
)
server <- function(input, output) {
    output$hist <- renderPlot(hist(rnorm(input$num)))
}
shinyApp(ui = ui, server = server)