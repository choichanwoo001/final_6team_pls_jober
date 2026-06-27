<template>
  <main class="home">
    <v-container class="py-8">
      <v-row justify="center">
        <v-col cols="12" lg="10">
          <header class="mb-6">
            <h1 class="text-h4 font-weight-bold mb-2">AI 사전 검증</h1>
            <p class="text-body-1 text-medium-emphasis mb-0">
              제출 전 반려 위험도와 보완 포인트를 확인합니다.
            </p>
          </header>

          <v-row>
            <v-col cols="12" md="7">
              <v-sheet border rounded class="pa-5">
                <v-textarea
                  v-model="content"
                  label="제출 내용"
                  rows="9"
                  auto-grow
                  variant="outlined"
                  hide-details="auto"
                />
                <v-text-field
                  v-model="question"
                  class="mt-4"
                  label="검증 질문"
                  variant="outlined"
                  hide-details="auto"
                />
                <v-slider
                  v-model="nResults"
                  class="mt-4"
                  label="유사 문서 수"
                  :min="1"
                  :max="10"
                  :step="1"
                  thumb-label
                  hide-details
                />
                <div class="d-flex align-center ga-3 mt-5">
                  <v-btn
                    color="primary"
                    :loading="isChecking"
                    :disabled="!canSubmit || isChecking"
                    @click="submitPrecheck"
                  >
                    검증 실행
                  </v-btn>
                  <v-chip v-if="result" :color="riskColor" variant="flat">
                    {{ result.approvalRisk }} · {{ result.score }}점
                  </v-chip>
                </div>
              </v-sheet>
            </v-col>

            <v-col cols="12" md="5">
              <v-alert
                v-if="errorMessage"
                type="error"
                variant="tonal"
                class="mb-4"
              >
                {{ errorMessage }}
              </v-alert>

              <v-sheet v-if="result" border rounded class="pa-5">
                <div class="d-flex align-center justify-space-between mb-4">
                  <div>
                    <div class="text-caption text-medium-emphasis">반려 위험도</div>
                    <div class="text-h5 font-weight-bold">{{ result.approvalRisk }}</div>
                  </div>
                  <v-progress-circular
                    :model-value="result.score"
                    :color="riskColor"
                    :size="72"
                    :width="8"
                  >
                    {{ result.score }}
                  </v-progress-circular>
                </div>

                <v-divider class="mb-4" />

                <h2 class="text-subtitle-1 font-weight-bold mb-2">판단 근거</h2>
                <v-list density="compact" class="mb-4">
                  <v-list-item
                    v-for="reason in result.reasons"
                    :key="reason"
                    :title="reason"
                  />
                </v-list>

                <h2 class="text-subtitle-1 font-weight-bold mb-2">수정 제안</h2>
                <v-list density="compact" class="mb-4">
                  <v-list-item
                    v-for="suggestion in result.suggestions"
                    :key="suggestion"
                    :title="suggestion"
                  />
                </v-list>

                <v-divider class="mb-4" />

                <div class="text-body-2 text-medium-emphasis">
                  총 {{ result.metrics.totalMs }}ms · 검색 {{ result.metrics.searchMs }}ms ·
                  감정 {{ result.metrics.sentimentMs }}ms · QA {{ result.metrics.qaMs }}ms
                  <span v-if="result.metrics.cacheHit"> · 캐시 사용</span>
                </div>
              </v-sheet>

              <v-sheet v-else border rounded class="pa-5 text-medium-emphasis">
                검증 결과가 여기에 표시됩니다.
              </v-sheet>
            </v-col>
          </v-row>
        </v-col>
      </v-row>
    </v-container>
  </main>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useMutation } from '@tanstack/vue-query'
import { precheckReview, type ReviewPrecheckResponse } from '../api/ai'

const content = ref('지원자는 관련 프로젝트 경험을 보유하고 있으며 요구 조건을 충족합니다.')
const question = ref('이 내용이 요구 조건을 충족하는가?')
const nResults = ref(5)
const result = ref<ReviewPrecheckResponse | null>(null)
const errorMessage = ref('')

const reviewMutation = useMutation({
  mutationFn: precheckReview,
  onSuccess(data) {
    result.value = data
    errorMessage.value = ''
  },
  onError(error) {
    result.value = null
    errorMessage.value = error instanceof Error
      ? '검증 시간이 지연되고 있습니다. 잠시 후 다시 시도해주세요.'
      : '검증 요청에 실패했습니다.'
  },
})

const canSubmit = computed(() => content.value.trim().length > 0 && question.value.trim().length > 0)
const isChecking = computed(() => reviewMutation.isPending.value)

const riskColor = computed(() => {
  if (!result.value) return 'grey'
  if (result.value.approvalRisk === 'LOW') return 'success'
  if (result.value.approvalRisk === 'MEDIUM') return 'warning'
  return 'error'
})

function submitPrecheck() {
  if (!canSubmit.value || isChecking.value) return
  reviewMutation.mutate({
    content: content.value,
    question: question.value,
    n_results: nResults.value,
  })
}
</script>

<style scoped>
.home {
  min-height: 100vh;
  background: #f7f9fb;
}
</style>
