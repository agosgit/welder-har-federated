// src/components/SensorChart.tsx
import React from 'react';
import { View } from 'react-native';
import { LineChart } from 'react-native-gifted-charts';

export default function SensorChart({
  data,
  color,
}: {
  data: number[];
  color: string;
}) {
  return (
    <View style={{ height: 70, marginVertical: 4 }}>
      <LineChart
        data={data.map((v, i) => ({ value: v, index: i }))}
        thickness={2}
        color={color}
        hideYAxisText
        hideDataPoints
        spacing={3}
        initialSpacing={0}
        animateOnDataChange={false}
        height={70}
      />
    </View>
  );
}
