{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit forms.test.session.endcriteria;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, Grids, Spin,
  bot.types, session.endcriteria;

type

  { TFormEndCriteriaTest }

  TFormEndCriteriaTest = class(TForm)
    ButtonBreakRandom: TButton;
    ButtonRandom: TButton;
    ButtonHit: TButton;
    ButtonMiss: TButton;
    ButtonTearDown: TButton;
    ButtonSetup: TButton;
    FloatSpinEditHitPorcentage: TFloatSpinEdit;
    MemoInfo: TMemo;
    MemoCurrentBlock: TMemo;
    MemoCurrentTrial: TMemo;
    StringGrid1: TStringGrid;
    procedure ButtonBreakRandomClick(Sender: TObject);
    procedure ButtonHitClick(Sender: TObject);
    procedure ButtonMissClick(Sender: TObject);
    procedure ButtonRandomClick(Sender: TObject);
    procedure ButtonTearDownClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure ButtonSetupClick(Sender: TObject);
  private
    FEndOfSession : Boolean;
    FMustBreakRandom : Boolean;
    FEndCriteria : TEndCriteria;
    procedure Play;
    procedure InvalidateCurrentBlock;
    procedure InvalidateCurrentTrial;
    procedure DoResponse(ATrialResult: TTrialResult);
    procedure Pass(Sender: TObject);
    procedure Fail(Sender: TObject);
    procedure EndSession;
    procedure Setup;
    procedure TearDown;
  public

  end;

var
  FormEndCriteriaTest: TFormEndCriteriaTest;

implementation

{$R *.lfm}

uses
  forms.main,
  session.constants.trials,
  session.configurationfile,
  session.fileutils,
  session.pool,
  sdl.app.trials.factory;

{ TFormEndCriteriaTest }

procedure TFormEndCriteriaTest.ButtonSetupClick(Sender: TObject);
begin
  StringGrid1.RowCount := 1;
  MemoCurrentBlock.Clear;
  MemoCurrentTrial.Clear;
  Setup;
  Play;
end;

procedure TFormEndCriteriaTest.FormCreate(Sender: TObject);
var
  j : integer;

  procedure AutoSizeCol(AColumn: integer);
  var
    i, W, WMax: integer;
  begin
    WMax := 0;
    with StringGrid1 do begin
      for i := 0 to (RowCount - 1) do begin
        W := Canvas.TextWidth(Cells[AColumn, i]);
        if W > WMax then
          WMax := W;
      end;
      ColWidths[AColumn] := WMax + 10;
    end;
  end;
begin
  with StringGrid1 do begin
    ColCount := 8;
    Cells[0, 0] := 'Session.Trial.UID';
    Cells[1, 0] := 'Session.Block.UID';
    Cells[2, 0] := 'Session.Block.Trial.UID';
    Cells[3, 0] := 'Session.Block.ID';
    Cells[4, 0] := 'Session.Block.Trial.ID';
    Cells[5, 0] := 'Session.Block.Name';
    Cells[6, 0] := 'Session.Block.Trial.Name';
    Cells[7, 0] := 'Result';

    for j := 0 to ColCount - 1 do begin
      AutoSizeCol(j);
    end;
  end;
end;

procedure TFormEndCriteriaTest.ButtonTearDownClick(Sender: TObject);
begin
  TearDown;
end;

procedure TFormEndCriteriaTest.ButtonHitClick(Sender: TObject);
begin
  if Assigned(FEndCriteria) then begin
    DoResponse(Hit);
  end;
end;

procedure TFormEndCriteriaTest.ButtonBreakRandomClick(Sender: TObject);
begin
  FMustBreakRandom := True;
end;

procedure TFormEndCriteriaTest.ButtonMissClick(Sender: TObject);
begin
  if Assigned(FEndCriteria) then begin
    DoResponse(Miss);
  end;
end;

procedure TFormEndCriteriaTest.ButtonRandomClick(Sender: TObject);
begin
  while not FEndOfSession do begin

    with FloatSpinEditHitPorcentage do begin
      if Random < (Value/1.0) then begin
        DoResponse(Hit);
      end else begin
        DoResponse(Miss);
      end;
    end;

    Application.ProcessMessages;
    if FMustBreakRandom then begin
      FMustBreakRandom := False;
      Break;
    end;
  end;
end;

procedure TFormEndCriteriaTest.Play;
begin
  if FEndCriteria.OfSession then begin
    EndSession;
  end else begin
    if FEndCriteria.OfBlock then begin
      FEndCriteria.InvalidateBlock;
      InvalidateCurrentBlock;
      Play;
    end else begin
      FEndCriteria.InvalidateTrial(ConfigurationFile.CurrentTrial);
      InvalidateCurrentTrial;
    end;
  end;
end;

procedure TFormEndCriteriaTest.InvalidateCurrentBlock;
begin
  MemoCurrentBlock.Append('-------------------------');
  MemoCurrentBlock.Append(ConfigurationFile.CurrentBlock.ToData);
end;

procedure TFormEndCriteriaTest.InvalidateCurrentTrial;
begin
  MemoCurrentTrial.Append('-------------------------');
  MemoCurrentTrial.Append(ConfigurationFile.CurrentTrial.ToData);
end;

procedure TFormEndCriteriaTest.Setup;
begin
  FEndOfSession := False;
  FMustBreakRandom := False;
  FormMain.CreateNewConfigurationFile;
  Pool.Counters.BeforeBeginSession;
  FEndCriteria := TEndCriteria.Create;
  FEndCriteria.OnHitCriteriaAtSessionEnd := @Pass;
  FEndCriteria.OnNotHitCriteriaAtSessionEnd := @Fail;
  FEndCriteria.InvalidateBlock;
  InvalidateCurrentBlock;
end;

procedure TFormEndCriteriaTest.TearDown;
begin
  Pool.Counters.BeforeEndSession;
  FEndCriteria.Free;
  FreeConfigurationFile;
end;

procedure TFormEndCriteriaTest.DoResponse(ATrialResult: TTrialResult);
var
  LLastResponse : string;
begin
  WriteStr(LLastResponse, ATrialResult);
  case ATrialResult of
    Hit: Pool.Counters.Hit;
    Miss: Pool.Counters.Miss;
    None: Pool.Counters.None;
  end;

  with StringGrid1 do begin
    InsertRowWithValues(RowCount, [
      (Pool.Session.Trial.UID + 1).ToString,
      (Pool.Session.Block.UID + 1).ToString,
      (Pool.Session.Block.Trial.UID + 1).ToString,
      (Pool.Session.Block.ID + 1).ToString,
      (Pool.Session.Block.Trial.ID + 1).ToString,
      ConfigurationFile.CurrentBlock.Name,
      ConfigurationFile.CurrentTrial.Parameters.Values[TrialKeys.NameKey],
      LLastResponse]);
  end;

  FEndCriteria.OfTrial;
  Play;
end;

procedure TFormEndCriteriaTest.Pass(Sender: TObject);
begin
  FEndOfSession := True;
  ShowMessage('Passed');
end;

procedure TFormEndCriteriaTest.Fail(Sender: TObject);
begin
  FEndOfSession := True;
  ShowMessage('Failed');
end;

procedure TFormEndCriteriaTest.EndSession;
begin
  FEndOfSession := True;
  TearDown;
end;

end.

