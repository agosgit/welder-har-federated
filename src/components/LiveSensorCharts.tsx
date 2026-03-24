// src/components/LiveSensorCharts.tsx
import React, { useEffect, useState } from "react";
import { View, Text } from "react-native";
import SkiaSensorChart from "./SkiaSensorChart";

type Vec3 = { x: number; y: number; z: number };

type LiveSensorChartsProps = {
  hz: { acc: number; gyr: number; mag: number };
  accDisp: Vec3;
  gyrDisp: Vec3;
  magDisp: Vec3;
};

type TripleArray = {
  x: number[];
  y: number[];
  z: number[];
};

export default function LiveSensorCharts({
  hz,
  accDisp,
  gyrDisp,
  magDisp,
}: LiveSensorChartsProps) {
  const [acc, setAcc] = useState<TripleArray>({ x: [], y: [], z: [] });
  const [gyr, setGyr] = useState<TripleArray>({ x: [], y: [], z: [] });
  const [mag, setMag] = useState<TripleArray>({ x: [], y: [], z: [] });

  const push = (arr: number[], v: number) => [...arr.slice(-99), v];

  useEffect(() => {
    setAcc(prev => ({
      x: push(prev.x, accDisp.x),
      y: push(prev.y, accDisp.y),
      z: push(prev.z, accDisp.z),
    }));
  }, [accDisp]);

  useEffect(() => {
    setGyr(prev => ({
      x: push(prev.x, gyrDisp.x),
      y: push(prev.y, gyrDisp.y),
      z: push(prev.z, gyrDisp.z),
    }));
  }, [gyrDisp]);

  useEffect(() => {
    setMag(prev => ({
      x: push(prev.x, magDisp.x),
      y: push(prev.y, magDisp.y),
      z: push(prev.z, magDisp.z),
    }));
  }, [magDisp]);

  return (
    <View style={{ paddingBottom: 12 }}>
      <Text>Accelerometer • {hz.acc.toFixed(1)} Hz</Text>
      <SkiaSensorChart xData={acc.x} yData={acc.y} zData={acc.z} />

      <Text style={{ marginTop: 10 }}>Gyroscope • {hz.gyr.toFixed(1)} Hz</Text>
      <SkiaSensorChart xData={gyr.x} yData={gyr.y} zData={gyr.z} />

      <Text style={{ marginTop: 10 }}>Magnetometer • {hz.mag.toFixed(1)} Hz</Text>
      <SkiaSensorChart xData={mag.x} yData={mag.y} zData={mag.z} />
    </View>
  );
}
