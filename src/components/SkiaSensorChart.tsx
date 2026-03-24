// src/components/SkiaSensorChart.tsx
import React from "react";
import { View } from "react-native";
import { Canvas, Path, Skia } from "@shopify/react-native-skia";

type SensorChartProps = {
  xData: number[];
  yData: number[];
  zData: number[];
  width?: number;
  height?: number;
  colorX?: string;
  colorY?: string;
  colorZ?: string;
};

export default function SkiaSensorChart({
  xData,
  yData,
  zData,
  width = 320,
  height = 100,
  colorX = "#ff3b30",
  colorY = "#0a84ff",
  colorZ = "#34c759",
}: SensorChartProps) {
  const makePath = (arr: number[]) => {
    const p = Skia.Path.Make();
    if (!arr.length) return p;

    const max = Math.max(...arr);
    const min = Math.min(...arr);
    const span = max - min || 1;

    arr.forEach((v, i) => {
      const x = (i / (arr.length - 1)) * width;
      const y = height - ((v - min) / span) * height;
      i === 0 ? p.moveTo(x, y) : p.lineTo(x, y);
    });

    return p;
  };

  return (
    <View style={{ width, height, marginVertical: 6 }}>
      <Canvas style={{ width, height }}>
        <Path path={makePath(xData)} color={colorX} style="stroke" strokeWidth={2} />
        <Path path={makePath(yData)} color={colorY} style="stroke" strokeWidth={2} />
        <Path path={makePath(zData)} color={colorZ} style="stroke" strokeWidth={2} />
      </Canvas>
    </View>
  );
}
