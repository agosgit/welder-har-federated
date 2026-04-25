import React, { useEffect, useState } from "react";
import { View, Text, FlatList, TouchableOpacity, Alert } from "react-native";
import { addLocalSample, getLocalSampleCount } from "../har/localDataset";
import { getCurrentModelVersion } from "../har/flClient";
import { type Window as MLWindow } from "../har/features";

interface FeedbackSectionProps {
  currentWindowData: MLWindow | null;
  onSaved?: () => void;
}

export default function FeedbackSection({ currentWindowData, onSaved }: FeedbackSectionProps) {
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  const [sampleCount, setSampleCount] = useState(0);

  const activityLabels = [
    "Duduk Aktif",
    "Berbaring Aktif",
    "Berdiri Aktif",
    "Berdiri Tidak Aktif",
    "Duduk Tidak Aktif",
    "Berbaring Tidak Aktif",
  ];

  async function refreshCount() {
    const count = await getLocalSampleCount();
    setSampleCount(count);
  }

  useEffect(() => {
    refreshCount();
  }, []);

  const handleSave = async () => {
    if (!selectedLabel) {
      Alert.alert("Pilih aktivitas", "Silakan pilih label aktivitas terlebih dahulu.");
      return;
    }

    if (!currentWindowData) {
      Alert.alert("Belum ada data", "Tunggu window sensor terbentuk dulu.");
      return;
    }

    try {
      await addLocalSample(
        currentWindowData,
        selectedLabel,
        getCurrentModelVersion()
      );

      await refreshCount();
      onSaved?.();

      Alert.alert("Sample tersimpan", `✅ Label ${selectedLabel} disimpan ke dataset lokal.`);
      setSelectedLabel(null);
    } catch (err: any) {
      Alert.alert("Gagal menyimpan", String(err?.message ?? err));
    }
  };

  return (
    <View style={{ marginTop: 16 }}>
      <Text style={{ fontWeight: "bold", marginBottom: 8 }}>Label Data Lokal</Text>

      <FlatList
        data={activityLabels}
        horizontal
        keyExtractor={(item) => item}
        renderItem={({ item }) => (
          <TouchableOpacity
            onPress={() => setSelectedLabel(item)}
            style={{
              backgroundColor: selectedLabel === item ? "#007AFF" : "#eee",
              paddingHorizontal: 12,
              paddingVertical: 8,
              borderRadius: 12,
              marginRight: 8,
            }}
          >
            <Text
              style={{
                color: selectedLabel === item ? "white" : "black",
                fontWeight: "600",
              }}
            >
              {item}
            </Text>
          </TouchableOpacity>
        )}
      />

      <TouchableOpacity
        onPress={handleSave}
        style={{
          marginTop: 12,
          backgroundColor: "#006D77",
          paddingVertical: 12,
          borderRadius: 12,
          alignItems: "center",
        }}
      >
        <Text style={{ color: "white", fontWeight: "700" }}>Simpan Sample Lokal</Text>
      </TouchableOpacity>

      <Text style={{ marginTop: 10, color: "#334155" }}>
        Total sample lokal: <Text style={{ fontWeight: "bold" }}>{sampleCount}</Text>
      </Text>
    </View>
  );
}