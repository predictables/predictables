from plotly.basedatatypes import BaseLayoutHierarchyType as _BaseLayoutHierarchyType
from typing import Any

class Data(_BaseLayoutHierarchyType):
    @property
    def barpolar(self): ...
    @barpolar.setter
    def barpolar(self, val) -> None: ...
    @property
    def bar(self): ...
    @bar.setter
    def bar(self, val) -> None: ...
    @property
    def box(self): ...
    @box.setter
    def box(self, val) -> None: ...
    @property
    def candlestick(self): ...
    @candlestick.setter
    def candlestick(self, val) -> None: ...
    @property
    def carpet(self): ...
    @carpet.setter
    def carpet(self, val) -> None: ...
    @property
    def choroplethmapbox(self): ...
    @choroplethmapbox.setter
    def choroplethmapbox(self, val) -> None: ...
    @property
    def choropleth(self): ...
    @choropleth.setter
    def choropleth(self, val) -> None: ...
    @property
    def cone(self): ...
    @cone.setter
    def cone(self, val) -> None: ...
    @property
    def contourcarpet(self): ...
    @contourcarpet.setter
    def contourcarpet(self, val) -> None: ...
    @property
    def contour(self): ...
    @contour.setter
    def contour(self, val) -> None: ...
    @property
    def densitymapbox(self): ...
    @densitymapbox.setter
    def densitymapbox(self, val) -> None: ...
    @property
    def funnelarea(self): ...
    @funnelarea.setter
    def funnelarea(self, val) -> None: ...
    @property
    def funnel(self): ...
    @funnel.setter
    def funnel(self, val) -> None: ...
    @property
    def heatmapgl(self): ...
    @heatmapgl.setter
    def heatmapgl(self, val) -> None: ...
    @property
    def heatmap(self): ...
    @heatmap.setter
    def heatmap(self, val) -> None: ...
    @property
    def histogram2dcontour(self): ...
    @histogram2dcontour.setter
    def histogram2dcontour(self, val) -> None: ...
    @property
    def histogram2d(self): ...
    @histogram2d.setter
    def histogram2d(self, val) -> None: ...
    @property
    def histogram(self): ...
    @histogram.setter
    def histogram(self, val) -> None: ...
    @property
    def icicle(self): ...
    @icicle.setter
    def icicle(self, val) -> None: ...
    @property
    def image(self): ...
    @image.setter
    def image(self, val) -> None: ...
    @property
    def indicator(self): ...
    @indicator.setter
    def indicator(self, val) -> None: ...
    @property
    def isosurface(self): ...
    @isosurface.setter
    def isosurface(self, val) -> None: ...
    @property
    def mesh3d(self): ...
    @mesh3d.setter
    def mesh3d(self, val) -> None: ...
    @property
    def ohlc(self): ...
    @ohlc.setter
    def ohlc(self, val) -> None: ...
    @property
    def parcats(self): ...
    @parcats.setter
    def parcats(self, val) -> None: ...
    @property
    def parcoords(self): ...
    @parcoords.setter
    def parcoords(self, val) -> None: ...
    @property
    def pie(self): ...
    @pie.setter
    def pie(self, val) -> None: ...
    @property
    def pointcloud(self): ...
    @pointcloud.setter
    def pointcloud(self, val) -> None: ...
    @property
    def sankey(self): ...
    @sankey.setter
    def sankey(self, val) -> None: ...
    @property
    def scatter3d(self): ...
    @scatter3d.setter
    def scatter3d(self, val) -> None: ...
    @property
    def scattercarpet(self): ...
    @scattercarpet.setter
    def scattercarpet(self, val) -> None: ...
    @property
    def scattergeo(self): ...
    @scattergeo.setter
    def scattergeo(self, val) -> None: ...
    @property
    def scattergl(self): ...
    @scattergl.setter
    def scattergl(self, val) -> None: ...
    @property
    def scattermapbox(self): ...
    @scattermapbox.setter
    def scattermapbox(self, val) -> None: ...
    @property
    def scatterpolargl(self): ...
    @scatterpolargl.setter
    def scatterpolargl(self, val) -> None: ...
    @property
    def scatterpolar(self): ...
    @scatterpolar.setter
    def scatterpolar(self, val) -> None: ...
    @property
    def scatter(self): ...
    @scatter.setter
    def scatter(self, val) -> None: ...
    @property
    def scattersmith(self): ...
    @scattersmith.setter
    def scattersmith(self, val) -> None: ...
    @property
    def scatterternary(self): ...
    @scatterternary.setter
    def scatterternary(self, val) -> None: ...
    @property
    def splom(self): ...
    @splom.setter
    def splom(self, val) -> None: ...
    @property
    def streamtube(self): ...
    @streamtube.setter
    def streamtube(self, val) -> None: ...
    @property
    def sunburst(self): ...
    @sunburst.setter
    def sunburst(self, val) -> None: ...
    @property
    def surface(self): ...
    @surface.setter
    def surface(self, val) -> None: ...
    @property
    def table(self): ...
    @table.setter
    def table(self, val) -> None: ...
    @property
    def treemap(self): ...
    @treemap.setter
    def treemap(self, val) -> None: ...
    @property
    def violin(self): ...
    @violin.setter
    def violin(self, val) -> None: ...
    @property
    def volume(self): ...
    @volume.setter
    def volume(self, val) -> None: ...
    @property
    def waterfall(self): ...
    @waterfall.setter
    def waterfall(self, val) -> None: ...
    def __init__(
        self,
        arg: Any | None = ...,
        barpolar: Any | None = ...,
        bar: Any | None = ...,
        box: Any | None = ...,
        candlestick: Any | None = ...,
        carpet: Any | None = ...,
        choroplethmapbox: Any | None = ...,
        choropleth: Any | None = ...,
        cone: Any | None = ...,
        contourcarpet: Any | None = ...,
        contour: Any | None = ...,
        densitymapbox: Any | None = ...,
        funnelarea: Any | None = ...,
        funnel: Any | None = ...,
        heatmapgl: Any | None = ...,
        heatmap: Any | None = ...,
        histogram2dcontour: Any | None = ...,
        histogram2d: Any | None = ...,
        histogram: Any | None = ...,
        icicle: Any | None = ...,
        image: Any | None = ...,
        indicator: Any | None = ...,
        isosurface: Any | None = ...,
        mesh3d: Any | None = ...,
        ohlc: Any | None = ...,
        parcats: Any | None = ...,
        parcoords: Any | None = ...,
        pie: Any | None = ...,
        pointcloud: Any | None = ...,
        sankey: Any | None = ...,
        scatter3d: Any | None = ...,
        scattercarpet: Any | None = ...,
        scattergeo: Any | None = ...,
        scattergl: Any | None = ...,
        scattermapbox: Any | None = ...,
        scatterpolargl: Any | None = ...,
        scatterpolar: Any | None = ...,
        scatter: Any | None = ...,
        scattersmith: Any | None = ...,
        scatterternary: Any | None = ...,
        splom: Any | None = ...,
        streamtube: Any | None = ...,
        sunburst: Any | None = ...,
        surface: Any | None = ...,
        table: Any | None = ...,
        treemap: Any | None = ...,
        violin: Any | None = ...,
        volume: Any | None = ...,
        waterfall: Any | None = ...,
        **kwargs
    ) -> None: ...