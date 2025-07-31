{
  Stimulus Control
  Copyright (C) 2023-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit Forms.Main;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, RTTICtrls, Forms, Controls, Graphics, Dialogs,
  StdCtrls, ExtCtrls, Schedules;

type

  { TForm1 }

  TForm1 = class(TForm)
    PanelOperandum: TPanel;
    ScheduleVR: TSchedule;
    procedure FormCreate(Sender: TObject);
    procedure PanelOperandumClick(Sender: TObject);
    procedure ConsequenceEvent(Sender: TObject);
    procedure ResponseEvent(Sender: TObject);
  end;

var
  Form1: TForm1;

implementation

{$R *.lfm}

{ TForm1 }

procedure TForm1.FormCreate(Sender: TObject);
begin
  ScheduleVR.Start;
end;

procedure TForm1.PanelOperandumClick(Sender: TObject);
begin
  ScheduleVR.DoResponse;
end;

procedure TForm1.ResponseEvent(Sender: TObject);
begin
  WriteLn(TSchedule(Sender).ComponentCount);
end;

procedure TForm1.ConsequenceEvent(Sender: TObject);
begin
  PanelOperandum.Color := RGBToColor(Random(256),Random(256),Random(256));
end;

end.

