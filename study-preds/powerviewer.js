var accessor = function(field) {
    return function(d) {
        return d[field];
    };
}

var projectX = function(rawData, xField) {
    // A transformer that projects CSV-type data onto a specific X-field
    // (column) for easier plotting with Plottable.js.
    var seriesMap = {};
    var result  = {
        series: [],
        seriesNames: [],
        xBounds: [Infinity, -Infinity],
        yBounds: [Infinity, -Infinity],
        xAccessor: accessor('x'),
        yAccessor: accessor('y'),
        nameAccessor: accessor('name'),
    };
    var xBounds = result.xBounds;
    var yBounds = result.yBounds;
    for (var i = 0, n = rawData.length; i < n; i++) {
        row = rawData[i];
        for (var field in row) {
            if (field !== xField && row.hasOwnProperty(field)) {
                var yVal = parseFloat(row[field]);
                var xVal = parseFloat(row[xField]);
                if (xVal < xBounds[0]) {
                    xBounds[0] = xVal;   // x minimum
                } else if (xVal > xBounds[1]) {
                    xBounds[1] = xVal;   // x maximum
                }
                if (yVal < yBounds[0]) {
                    yBounds[0] = yVal;   // y minimum
                } else if (yVal > yBounds[1]) {
                    yBounds[1] = yVal;   // y maximum
                }
                seriesRow = {'name': field, 'x': xVal, 'y': yVal};
                if (seriesMap.hasOwnProperty(field)) {
                    seriesMap[field].push(seriesRow)
                } else {
                    seriesMap[field] = [seriesRow];
                }
            }
        }
    }
    var seriesNames = result.seriesNames;
    var series = result.series;
    for (var field in seriesMap) {
        if (seriesMap.hasOwnProperty(field)) {
            seriesNames.push(field);
            series.push(seriesMap[field])
        }
    }
    return result;
}

function drawPlot(dataFile, xField, svgElementId) {
    d3.csv(dataFile, function(rawData) {

        var data = projectX(rawData, xField);

        var xScale = new Plottable.Scales.Linear();
        var yScale = new Plottable.Scales.Linear();
        yScale.domain([0.4, 1.0]);

        var colorScale = new Plottable.Scales.Color();
        colorScale.domain(data.seriesNames);

        var legend = new Plottable.Components.Legend(colorScale);
        legend.maxEntriesPerRow(Infinity);

        var linePlot = new Plottable.Plots.Line()
            .x(data.xAccessor, xScale)
            .y(data.yAccessor, yScale)
            .attr("stroke-width", 3)
            .attr("stroke", data.nameAccessor, colorScale);

        var chancePlot = new Plottable.Plots.Line()
            .x(data.xAccessor, xScale)
            .y(function (d) { return 0.5; }, yScale)
            .attr("stroke-width", 1)
            .attr("stroke", "#dddddd")
            .attr("stroke-dasharray", 4);

        data.series.forEach(function(d) {
            var dataSet = new Plottable.Dataset(d, {name: d[0].name});
            linePlot.addDataset(dataSet);
            chancePlot.addDataset(dataSet);
        });

        var plots = new Plottable.Components.Group([
                linePlot,
                chancePlot
        ]);

        var xAxis = new Plottable.Axes.Numeric(xScale, "bottom");
        var yAxis = new Plottable.Axes.Numeric(yScale, "left");

        var title = new Plottable.Components.TitleLabel("Resolving power (AUC) vs. " + xField, "0");
        var xLabel = new Plottable.Components.AxisLabel(xField, "0");
        var yLabel = new Plottable.Components.AxisLabel("AUC", "270");

        var table = new Plottable.Components.Table([
                [null,   null,  title, null],
                [yLabel, yAxis, plots, legend],
                [null,   null,  xAxis, null],
                [null,   null,  xLabel, null]
        ]);

        table.renderTo(svgElementId);

        var adjustOpacity = function(plot, legendText) {
            plot.attr("opacity", function(d, i, ds) {
                return ds.metadata().name === legendText ? 1 : .05;
            });
        }

        new Plottable.Interactions.Click()
            .attachTo(legend)
            .onClick(function(p) {
                if (legend.entitiesAt(p)[0] !== undefined) {
                    var selected = legend.entitiesAt(p)[0].datum;
                    adjustOpacity(linePlot, selected);
                }
            });

        new Plottable.Interactions.Click()
            .attachTo(plots)
            .onClick(function() {
                linePlot.attr("opacity", 1);
            });
    });
}
