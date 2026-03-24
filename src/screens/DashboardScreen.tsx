// screens/DashboardScreen.tsx
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Modal,
} from "react-native";
import { LineChart } from "react-native-gifted-charts";
import axios from "axios";

type TimelineRec = {
  ts: number;
  label: string;
  motionVar: number;
  samples: number;
};
type Welder = { id: string; name: string };

type DashboardApi = {
  mode: "dummy" | "real";
  welders: Welder[];
  managementData: Record<string, TimelineRec[]>;
  timestamp: string;
};

export default function DashboardScreen() {
  const [mode, setMode] = useState("dummy");
  const [welders, setWelders] = useState<Welder[]>([]);
  const [managementData, setManagementData] = useState<Record<string, TimelineRec[]>>({});
  const [produktivitas, setProduktivitas] = useState(0);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [selectedWelder, setSelectedWelder] = useState<Welder | null>(null);

  useEffect(() => {
    axios
      .get<DashboardApi>("http://10.145.109.150:8082/api/dashboard?seed=573")
      .then((res) => {
        const data = res.data;
        setMode(data.mode);
        setWelders(data.welders);
        setManagementData(data.managementData);

        // Hitung produktivitas & alert
        const all = Object.values(data.managementData).flat();
        const active = all.filter((d) =>
          ["Lying active", "Sitting active", "Standing active"].includes(d.label)
        ).length;
        const total = all.length;
        setProduktivitas(Math.round((active / total) * 100));

        // Ambil data unsafe terakhir
        const now = Date.now();
        const fiveMinAgo = now - 5 * 60 * 1000;
        // Ambil data unsafe saat ini (bukan 5 menit terakhir)
        const unsafeNow = [];
        for (const [id, arr] of Object.entries(data.managementData)) {
          const last = arr[arr.length - 1];
          if (last?.label === "Unsafe condition") {
            unsafeNow.push({ welderId: id, rec: last });
          }
        }
        setAlerts(unsafeNow);

      })
      .catch((err) => console.log("❌ Error ambil data:", err.message));
  }, []);

  // 🔹 Siapkan tren aktivitas sinkron antar welder (tanpa crash)
  let trend: { label: string; active: number; idle: number; unsafe: number }[] = [];

  const welderIds = Object.keys(managementData);
  if (welderIds.length > 0) {
    const firstWelderId = welderIds[0];
    const firstTimeline = managementData[firstWelderId] || [];
    const totalSteps = Math.min(firstTimeline.length, 200); // batasi biar gak berat

    trend = Array.from({ length: totalSteps }).map((_, i) => {
      let active = 0,
        idle = 0,
        unsafe = 0;

      for (const id of welderIds) {
        const rec = managementData[id]?.[i];
        if (!rec) continue;
        if (["Lying active", "Sitting active", "Standing active"].includes(rec.label)) active++;
        else if (["Lying inactive", "Sitting inactive", "Standing inactive"].includes(rec.label))
          idle++;
        else unsafe++;
      }

      return { label: `T${i + 1}`, active, idle, unsafe };
    });
  }

  const activeData = trend.map((d) => ({ value: d.active, label: d.label }));
  const idleData = trend.map((d) => ({ value: d.idle, label: d.label }));
  const unsafeData = trend.map((d) => ({ value: d.unsafe, label: d.label }));

  const chartData = [
    { data: activeData, color: "green", thickness: 2 },
    { data: idleData, color: "orange", thickness: 2 },
    { data: unsafeData, color: "red", thickness: 2 },
  ];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>HAR Welder Dashboard</Text>
      {/* <View style={[styles.modeBadge, mode === "real" ? styles.modeReal : styles.modeDummy]}>
        <Text style={styles.modeText}>
          Mode: {mode === "real" ? "Real Data" : "Dummy Data"}
        </Text>
      </View> */}

      {/* KPI cards */}
      <View style={styles.kpiContainer}>
        <View style={styles.kpiCard}>
          <Text style={styles.kpiLabel}>Welders Online</Text>
          <Text style={styles.kpiValue}>{welders.length}</Text>
        </View>
        <View style={styles.kpiCard}>
          <Text style={styles.kpiLabel}>Produktivitas</Text>
          <Text style={styles.kpiValue}>{produktivitas}%</Text>
        </View>
        <View style={styles.kpiCard}>
          <Text style={styles.kpiLabel}>Unsafe</Text>
          <Text style={[styles.kpiValue, { color: "red" }]}>
            {alerts.length}
          </Text>
        </View>
      </View>

      {/* Chart */}
      <View style={styles.card}>
  <Text style={styles.chartTitle}>Activity Over Time</Text>

  {trend.length > 0 ? (
    <View style={{ alignItems: "center" }}>
      {/* Ambil waktu real untuk sumbu X */}
      <LineChart
        data={trend.map((d) => ({ value: d.active }))}
        dataSet={[
          { data: trend.map((d) => ({ value: d.idle })), color: "orange", thickness: 2 },
          { data: trend.map((d) => ({ value: d.unsafe })), color: "red", thickness: 2 },
          { data: trend.map((d) => ({ value: d.active })), color: "green", thickness: 3 },
        ]}
        height={250}
        curved
        areaChart={false}
        thickness={2}
        spacing={50}
        initialSpacing={10}
        yAxisTextStyle={{ color: "#555" }}
        xAxisLabelTextStyle={{ color: "#555", fontSize: 10 }}
        noOfSections={4}
        hideDataPoints={false}
        dataPointsHeight={3}
        dataPointsWidth={3}
        showVerticalLines
        rulesColor="#ddd"
        xAxisLabelTexts={(() => {
          // ambil welder pertama buat acuan waktu
          const firstId = Object.keys(managementData)[0];
          const timeline = managementData[firstId] || [];
          // tampilkan label tiap 30 titik biar gak nabrak
          return timeline.map((rec, i) =>
            i % 30 === 0
              ? new Date(rec.ts).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })
              : ""
          );
        })()}
      />

      {/* Label sumbu X */}
      {/* <Text style={{ marginTop: 6, color: "#666" }}>Time (HH:MM)</Text> */}
      <Text style={{ marginTop: 14, color: "#666" }}>Time (HH:MM)</Text>

    </View>
  ) : (
    <Text style={{ textAlign: "center", color: "#888", marginTop: 10 }}>
      Tidak ada data grafik.
    </Text>
  )}

  {/* Label sumbu Y di luar chart */}
  <View
    style={{
      position: "absolute",
      top: 130,
      left: -45,
      transform: [{ rotate: "-90deg" }],
    }}
  >
    <Text style={{ color: "#666" }}>Number of Welders</Text>
  </View>

  {/* Legend */}
  <View style={styles.legendContainer}>
    <View style={styles.legendItem}>
      <View style={[styles.legendDot, { backgroundColor: "green" }]} />
      <Text>Active</Text>
    </View>
    <View style={styles.legendItem}>
      <View style={[styles.legendDot, { backgroundColor: "orange" }]} />
      <Text>Idle</Text>
    </View>
    <View style={styles.legendItem}>
      <View style={[styles.legendDot, { backgroundColor: "red" }]} />
      <Text>Unsafe</Text>
    </View>
  </View>
</View>



      {/* Alerts */}
      <View style={styles.card}>
        <Text style={styles.chartTitle}>Recent Unsafe Alerts</Text>
        {alerts.length === 0 ? (
          <Text style={{ color: "#888" }}>No unsafe posture detected.</Text>
        ) : (
          alerts.map((a, i) => (
            <View key={i} style={styles.alertItem}>
              <Text>⚠️ {a.welderId}</Text>
              <Text style={{ color: "#888" }}>
                at {new Date(a.rec.ts).toLocaleTimeString()}
              </Text>
            </View>
          ))
        )}
      </View>

      {/* Per welder */}
      <View style={styles.card}>
        <Text style={styles.chartTitle}>Per-Welder Status (Real-time)</Text>
        {welders.map((item) => {
  const last = managementData[item.id]?.slice(-1)[0];
  const label = last?.label || "-";
  return (
    <View key={item.id} style={styles.welderRow}>
      <Text style={{ flex: 1 }}>{item.name}</Text>
      <Text style={{ flex: 1, color: "#555" }}>
        {new Date(last?.ts || 0).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        })}
      </Text>
      <Text style={{ flex: 1 }}>{label}</Text>
      <TouchableOpacity
        onPress={() => setSelectedWelder(item)}
        style={styles.viewButton}
      >
        <Text style={{ color: "white" }}>View</Text>
      </TouchableOpacity>
    </View>
  );
})}

      </View>

      {/* Modal Timeline */}
      <Modal visible={!!selectedWelder} animationType="slide">
        <ScrollView style={{ flex: 1, padding: 16 }}>
        {welders.map((item) => {
          const last = managementData[item.id]?.slice(-1)[0];
          const label = last?.label || "-";
          return (
            <View key={item.id} style={styles.welderRow}>
              <Text style={{ flex: 1 }}>{item.name}</Text>
              <Text style={{ flex: 1, color: "#555" }}>
                {new Date(last?.ts || 0).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </Text>
              <Text style={{ flex: 1 }}>{label}</Text>
              <TouchableOpacity
                onPress={() => setSelectedWelder(item)}
                style={styles.viewButton}
              >
                <Text style={{ color: "white" }}>View</Text>
              </TouchableOpacity>
            </View>
          );
        })}

        </ScrollView>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f9fafb", padding: 16 },
  title: { fontSize: 22, fontWeight: "bold", marginBottom: 10 },
  modeBadge: { alignSelf: "flex-start", padding: 6, borderRadius: 8 },
  modeDummy: { backgroundColor: "#fde68a" },
  modeReal: { backgroundColor: "#a7f3d0" },
  modeText: { fontWeight: "600" },
  kpiContainer: { flexDirection: "row", justifyContent: "space-between", marginVertical: 10 },
  kpiCard: {
    backgroundColor: "white",
    flex: 1,
    marginHorizontal: 4,
    padding: 10,
    borderRadius: 10,
    alignItems: "center",
    elevation: 2,
  },
  kpiLabel: { color: "#555" },
  kpiValue: { fontSize: 22, fontWeight: "bold" },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    elevation: 2,
  },
  chartTitle: { fontWeight: "bold", marginBottom: 8 },
  legendContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginTop: 10,
  },
  legendItem: { flexDirection: "row", alignItems: "center" },
  legendDot: { width: 10, height: 10, borderRadius: 5, marginRight: 6 },
  alertItem: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 4 },
  welderRow: { flexDirection: "row", alignItems: "center", paddingVertical: 6 },
  viewButton: {
    backgroundColor: "#0ea5e9",
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 6,
  },
});
