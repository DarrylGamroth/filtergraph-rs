//! Conformance tests for the transport-independent algorithm interface.

use core::convert::Infallible;
use core::mem::size_of;
use core::ops::Range;

use filtergraph_algorithms::{
    AlgorithmPlan, InputProgress, NoWorkspace, OutputProgress, ProgressResult,
};

#[derive(Debug, Eq, PartialEq)]
enum TestError {
    Active,
    Inactive,
    Incomplete,
    InvalidRange,
    WrongExtent,
}

#[derive(Debug)]
struct AddOnePlan;

impl AlgorithmPlan for AddOnePlan {
    type Input<'a> = &'a [i32];
    type Output<'a> = &'a mut [i32];
    type Workspace = NoWorkspace;
    type Error = TestError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if output.len() != input.len() {
            return Err(TestError::WrongExtent);
        }

        for (output, input) in output.iter_mut().zip(input) {
            *output = *input + 1;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ProgressiveAddOnePlan {
    extent: usize,
}

#[derive(Debug, Default)]
struct ProgressiveWorkspace {
    active: bool,
    processed: usize,
}

impl AlgorithmPlan for ProgressiveAddOnePlan {
    type Input<'a> = &'a [i32];
    type Output<'a> = &'a mut [i32];
    type Workspace = ProgressiveWorkspace;
    type Error = TestError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        _workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        check_extent(self.extent, output, input)?;
        for (output, input) in output.iter_mut().zip(input) {
            *output = *input + 1;
        }
        Ok(())
    }
}

impl InputProgress for ProgressiveAddOnePlan {
    fn start(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        if workspace.active {
            return Err(TestError::Active);
        }
        workspace.active = true;
        workspace.processed = 0;
        Ok(())
    }

    fn process_range<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
        workrange: Range<usize>,
    ) -> Result<ProgressResult, Self::Error> {
        if !workspace.active {
            return Err(TestError::Inactive);
        }
        check_extent(self.extent, output, input)?;
        if workrange.is_empty()
            || workrange.start != workspace.processed
            || workrange.end > self.extent
        {
            return Err(TestError::InvalidRange);
        }

        for index in workrange.clone() {
            output[index] = input[index] + 1;
        }
        workspace.processed = workrange.end;
        Ok(ProgressResult::new(workrange.end))
    }

    fn finish<'output>(
        &self,
        _output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
    ) -> Result<(), Self::Error> {
        if !workspace.active {
            return Err(TestError::Inactive);
        }
        if workspace.processed != self.extent {
            return Err(TestError::Incomplete);
        }
        workspace.active = false;
        Ok(())
    }

    fn abort(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        if !workspace.active {
            return Err(TestError::Inactive);
        }
        workspace.active = false;
        workspace.processed = 0;
        Ok(())
    }
}

impl OutputProgress for ProgressiveAddOnePlan {}

#[derive(Debug)]
struct StatefulSumPlan {
    extent: usize,
}

#[derive(Debug)]
struct StatefulSumWorkspace {
    active: bool,
    processed: usize,
    state: i32,
    next_state: i32,
}

impl AlgorithmPlan for StatefulSumPlan {
    type Input<'a> = &'a [i32];
    type Output<'a> = &'a mut i32;
    type Workspace = StatefulSumWorkspace;
    type Error = TestError;

    fn process<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
    ) -> Result<(), Self::Error> {
        if input.len() != self.extent {
            return Err(TestError::WrongExtent);
        }
        let next_state = workspace.state + input.iter().sum::<i32>();
        *output = next_state;
        workspace.state = next_state;
        workspace.next_state = next_state;
        Ok(())
    }
}

impl InputProgress for StatefulSumPlan {
    fn start(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        if workspace.active {
            return Err(TestError::Active);
        }
        workspace.active = true;
        workspace.processed = 0;
        workspace.next_state = workspace.state;
        Ok(())
    }

    fn process_range<'input, 'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
        input: Self::Input<'input>,
        workrange: Range<usize>,
    ) -> Result<ProgressResult, Self::Error> {
        if !workspace.active {
            return Err(TestError::Inactive);
        }
        if input.len() != self.extent {
            return Err(TestError::WrongExtent);
        }
        if workrange.is_empty()
            || workrange.start != workspace.processed
            || workrange.end > self.extent
        {
            return Err(TestError::InvalidRange);
        }

        workspace.next_state += input[workrange.clone()].iter().sum::<i32>();
        workspace.processed = workrange.end;
        *output = workspace.next_state;
        Ok(ProgressResult::default())
    }

    fn finish<'output>(
        &self,
        output: Self::Output<'output>,
        workspace: &mut Self::Workspace,
    ) -> Result<(), Self::Error> {
        if !workspace.active {
            return Err(TestError::Inactive);
        }
        if workspace.processed != self.extent {
            return Err(TestError::Incomplete);
        }
        workspace.state = workspace.next_state;
        workspace.active = false;
        *output = workspace.state;
        Ok(())
    }

    fn abort(&self, workspace: &mut Self::Workspace) -> Result<(), Self::Error> {
        if !workspace.active {
            return Err(TestError::Inactive);
        }
        workspace.next_state = workspace.state;
        workspace.processed = 0;
        workspace.active = false;
        Ok(())
    }
}

fn check_extent(extent: usize, output: &[i32], input: &[i32]) -> Result<(), TestError> {
    if input.len() == extent && output.len() == extent {
        Ok(())
    } else {
        Err(TestError::WrongExtent)
    }
}

fn require_output_progress<P: OutputProgress>(_plan: &P) {}

#[test]
fn progress_result_identifies_an_immutable_prefix() {
    assert!(ProgressResult::default().is_empty());
    assert_eq!(ProgressResult::new(12).len(), 12);
    assert!(!ProgressResult::new(12).is_empty());
    assert_eq!(size_of::<ProgressResult>(), size_of::<usize>());
}

#[test]
fn no_workspace_is_zero_size() {
    assert_eq!(size_of::<NoWorkspace>(), 0);
}

#[test]
fn complete_processing_uses_borrowed_storage() {
    let plan = AddOnePlan;
    let input = [1, 2, 3];
    let mut output = [0; 3];
    let mut workspace = NoWorkspace;

    plan.process(&mut output, &mut workspace, &input).unwrap();

    assert_eq!(output, [2, 3, 4]);
}

#[test]
fn progressive_output_is_a_static_capability() {
    let plan = ProgressiveAddOnePlan { extent: 4 };
    require_output_progress(&plan);

    let input = [1, 2, 3, 4];
    let mut output = [0; 4];
    let mut workspace = ProgressiveWorkspace::default();

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &input, 0..2),
        Ok(ProgressResult::new(2))
    );
    assert_eq!(output, [2, 3, 0, 0]);
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &input, 2..4),
        Ok(ProgressResult::new(4))
    );
    plan.finish(&mut output, &mut workspace).unwrap();
    assert_eq!(output, [2, 3, 4, 5]);
}

#[test]
fn abort_preserves_committed_mathematical_history() {
    let plan = StatefulSumPlan { extent: 3 };
    let input = [1, 2, 3];
    let mut output = 0;
    let mut workspace = StatefulSumWorkspace {
        active: false,
        processed: 0,
        state: 10,
        next_state: 10,
    };

    plan.start(&mut workspace).unwrap();
    assert_eq!(
        plan.process_range(&mut output, &mut workspace, &input, 0..2),
        Ok(ProgressResult::default())
    );
    assert_eq!(output, 13);
    assert_eq!(workspace.state, 10);
    plan.abort(&mut workspace).unwrap();
    assert_eq!(workspace.state, 10);
    assert_eq!(workspace.next_state, 10);

    plan.start(&mut workspace).unwrap();
    let _ = plan
        .process_range(&mut output, &mut workspace, &input, 0..3)
        .unwrap();
    plan.finish(&mut output, &mut workspace).unwrap();
    assert_eq!(output, 16);
    assert_eq!(workspace.state, 16);
}

#[test]
fn result_error_can_be_uninhabited() {
    struct InfalliblePlan;

    impl AlgorithmPlan for InfalliblePlan {
        type Input<'a> = &'a ();
        type Output<'a> = &'a mut ();
        type Workspace = NoWorkspace;
        type Error = Infallible;

        fn process<'input, 'output>(
            &self,
            _output: Self::Output<'output>,
            _workspace: &mut Self::Workspace,
            _input: Self::Input<'input>,
        ) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    InfalliblePlan
        .process(&mut (), &mut NoWorkspace, &())
        .unwrap();
}
