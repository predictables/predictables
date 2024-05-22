from bokeh.plotting import figure
import streamlit as st

## -- Simple line plot ---------------------
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# Create the figure
p = figure(title="Simple Line Plot Example", x_axis_label="x", y_axis_label="y")

# Add a line renderer
p.line(x, y, legend_label="Temp.", line_width=2)

# Show the plot
st.bokeh_chart(p)

## -- Multiple line plot ---------------------
y1 = [6, 7, 2, 4, 5]
y2 = [2, 3, 4, 5, 6]
y3 = [4, 5, 6, 7, 8]

p = figure(title="Multiple Line Plot Example", x_axis_label="x", y_axis_label="y")
p.line(x, y1, legend_label="Temp-1", line_width=2, color="red")
p.line(x, y2, legend_label="Temp-2", line_width=2, color="blue")
p.line(x, y3, legend_label="Temp-3", line_width=2, color="green")

st.bokeh_chart(p)

## -- Multiple Glyphs -------------------------
p = figure(title="Multiple Glyphs Example", x_axis_label="x", y_axis_label="y")
p.line(x, y1, legend_label="Line 1", line_width=2, color="red")
p.line(x, y2, legend_label="Line 2", line_width=2, color="blue")
p.scatter(x, y3, legend_label="Scatter", color="green", size=12)

st.bokeh_chart(p)

## -- Bars --------------------------
p = figure(title="Bars Example", x_axis_label="x", y_axis_label="y")
p.line(x, y1, legend_label="Line", line_width=2, color="red")
p.vbar(x=x, top=y2, legend_label="Bar", width=0.5, bottom=0, color="blue")

st.bokeh_chart(p)

## -- Circles --------------------------
y4 = [3, 6, 2, 7, 5]
p = figure(title="Circles Example", x_axis_label="x", y_axis_label="y")
p.line(x, y1, legend_label="Line", line_width=2, color="red")
p.vbar(x=x, top=y2, legend_label="Bar", width=0.5, bottom=0, color="blue")
p.scatter(
    x,
    y3,
    marker="circle",
    size=80,
    legend_label="big circle",
    fill_color="green",
    line_color="black",
    fill_alpha=0.5,
)
p.scatter(x, y4, legend_label="small scatter", color="green", size=12)

st.bokeh_chart(p)

## -- Adding styling and a legend --------------------------
p = figure(
    title="Styling and Legend Example -- on the right side!",
    x_axis_label="x",
    y_axis_label="y",
)
line = p.line(x, y1, legend_label="Line", line_width=2, color="red")
bar = p.vbar(
    x=x, top=y2, legend_label="Bar", width=0.5, bottom=0, color="blue", fill_alpha=0.5
)
circle = p.scatter(
    x,
    y4,
    marker="circle",
    size=80,
    legend_label="big circle",
    fill_color="green",
    line_color="black",
    fill_alpha=0.5,
)

p.legend.location = "top_left"
p.legend.click_policy = "hide"
p.legend.title = "This is a legend"

p.title_location = "right"


st.bokeh_chart(p)