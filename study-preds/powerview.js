function accessor(field) {
    // returns accessor (item getter) function
    return function(d) {
        return d[field];
    };
}

function sorted(arr) {
    // sorts array without mutating
    return arr.slice(0).sort();
}

function indexMap(arr) {
    // given an array, map values to their indices
    var result = {};
    for (var i = 0, n = arr.length; i < n; i++) {
        result[arr[i]] = i;
    }
    return result;
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

function drawPlot(dataFile, xField, title, containerId) {
    d3.csv(dataFile, function(rawData) {

        var data = projectX(rawData, xField);

        var xScale = new Plottable.Scales.Linear();
        var yScale = new Plottable.Scales.Linear();
        yScale.domain([0.45, 1.05]);

        var colorScale = new Plottable.Scales.Color();
        colorScale.domain(sorted(data.seriesNames));

        var order = indexMap(data.seriesNames);

        var legend = new Plottable.Components.Legend(colorScale);
        legend.maxEntriesPerRow(1);
        legend.comparator(
            function (a, b) {
                return order[a] - order[b];
            }
        );

        var linePlot = new Plottable.Plots.Line()
            .x(data.xAccessor, xScale)
            .y(data.yAccessor, yScale)
            .attr("stroke-width", 3)
            .attr("stroke", data.nameAccessor, colorScale);

        var chancePlot = new Plottable.Plots.Line()
            .x(data.xAccessor, xScale)
            .y(function (d) { return 0.5; }, yScale)
            .attr("stroke-width", 2)
            .attr("stroke", "#dddddd")
            .attr("stroke-dasharray", 4);

        data.series.forEach(function(d) {
            var dataSet = new Plottable.Dataset(d, {name: d[0].name});
            linePlot.addDataset(dataSet);
            chancePlot.addDataset(dataSet);
        });

        var plots = new Plottable.Components.Group([
                chancePlot,
                linePlot
        ]);

        var xAxis = new Plottable.Axes.Numeric(xScale, "bottom");
        var yAxis = new Plottable.Axes.Numeric(yScale, "left");

        var titleLabel = new Plottable.Components.TitleLabel(title, "0");
        var xLabel = new Plottable.Components.AxisLabel(xField, "0");
        var yLabel = new Plottable.Components.AxisLabel("Resolving power (AUC)", "270");

        var table = new Plottable.Components.Table([
                [null,   null,  titleLabel, null   ],
                [yLabel, yAxis, plots,      legend ],
                [null,   null,  xAxis,      null   ],
                [null,   null,  xLabel,     null   ]
        ]);

        var svg = d3.select(containerId)
            .append("div")
            .attr("style", "display:block; height:640px;")
            .append("svg")
            .attr("width", "100%")
            .attr("height", "600px")
            ;

        table.renderTo(svg);

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
